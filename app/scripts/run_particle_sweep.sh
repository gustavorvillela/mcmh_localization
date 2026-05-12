#!/bin/bash
# run_particle_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
# Executa variações de partículas para cada cenário (C/M/A) e bag.
#
# Uso:
#   ./run_particle_sweep.sh                     # todos os bags, todos os cenários
#   ./run_particle_sweep.sh L_rest.bag          # só esse bag, todos os cenários
#   ./run_particle_sweep.sh --scenario C        # todos os bags, só cenário C
#   ./run_particle_sweep.sh L_rest.bag --scenario M
#
# Estrutura de saída:
#   results/
#     C/                  ← conservador
#       house/
#         house_MCL_1000p_run1.txt
#         poses_house_MCL_1000p_run1.txt
#     M/                  ← médio
#     A/                  ← agressivo
# ─────────────────────────────────────────────────────────────────────────────

MODES=("MCL" "AMCL")
PARTICLE_COUNTS=(1000)
REPEATS=1

PKG_DIR="$(rospack find mcmh_localization)"
DEFAULT_BAG_DIR="$PKG_DIR/bags"
BASE_RESULTS_DIR="$PKG_DIR/results"
PARAMS_DIR="$PKG_DIR/params"

# ── Mapa de cenário → yaml ────────────────────────────────────────────────
declare -A SCENARIO_YAML
SCENARIO_YAML["C"]="$PARAMS_DIR/amhmcl_conservative.yaml"
SCENARIO_YAML["M"]="$PARAMS_DIR/amhmcl_medium.yaml"
SCENARIO_YAML["A"]="$PARAMS_DIR/amhmcl_aggressive.yaml"

ALL_SCENARIOS=("C" "M" "A")

# ── Parse de argumentos ───────────────────────────────────────────────────
SELECTED_SCENARIOS=()
BAG_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            shift
            SELECTED_SCENARIOS+=("$1")
            ;;
        *)
            BAG_ARGS+=("$1")
            ;;
    esac
    shift
done

# Se nenhum cenário especificado, roda todos
if [ ${#SELECTED_SCENARIOS[@]} -eq 0 ]; then
    SELECTED_SCENARIOS=("${ALL_SCENARIOS[@]}")
fi

# ── Determina bags ────────────────────────────────────────────────────────
if [ ${#BAG_ARGS[@]} -eq 0 ]; then
    BAGS=("$DEFAULT_BAG_DIR"/*.bag)
else
    BAGS=()
    for ARG in "${BAG_ARGS[@]}"; do
        if [ ! -f "$ARG" ] && [ ! -d "$ARG" ]; then
            ARG="$DEFAULT_BAG_DIR/$ARG"
        fi
        if [ -f "$ARG" ]; then
            BAGS+=("$ARG")
        elif [ -d "$ARG" ]; then
            for BAG_FILE in "$ARG"/*.bag; do
                [ -e "$BAG_FILE" ] && BAGS+=("$BAG_FILE")
            done
        else
            echo "Aviso: argumento inválido ($ARG), ignorado."
        fi
    done
    if [ ${#BAGS[@]} -eq 0 ]; then
        echo "Erro: nenhum arquivo .bag válido encontrado."
        exit 1
    fi
fi

# ── Limpeza seletiva por cenário ──────────────────────────────────────────
echo "=== Limpando resultados anteriores dos cenários selecionados ==="
for SCENARIO in "${SELECTED_SCENARIOS[@]}"; do
    SCENARIO_DIR="$BASE_RESULTS_DIR/$SCENARIO"
    if [ -d "$SCENARIO_DIR" ]; then
        echo "  Limpando $SCENARIO_DIR ..."
        find "$SCENARIO_DIR" -type f \( -name "*.txt" -o -name "*.html" -o -name "*.png" \) -delete
    fi
    mkdir -p "$SCENARIO_DIR"
done

# ── roscore ───────────────────────────────────────────────────────────────
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost

ROSCORE_PID=""
if ! pgrep -f roscore > /dev/null; then
    echo "Iniciando roscore..."
    roscore &
    ROSCORE_PID=$!
fi

echo "Aguardando roscore..."
until rostopic list >/dev/null 2>&1; do sleep 1; done
echo "roscore pronto!"

# ── Loop principal ────────────────────────────────────────────────────────
for SCENARIO in "${SELECTED_SCENARIOS[@]}"; do

    YAML_FILE="${SCENARIO_YAML[$SCENARIO]}"

    if [ ! -f "$YAML_FILE" ]; then
        echo "ERRO: YAML não encontrado para cenário $SCENARIO: $YAML_FILE"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  CENÁRIO: $SCENARIO  →  $(basename $YAML_FILE)"
    echo "╚══════════════════════════════════════════════════╝"

    # Carrega parâmetros do cenário
    rosparam load "$YAML_FILE"

    for BAG in "${BAGS[@]}"; do
        BAG_NAME=$(basename "$BAG" .bag)
        RESULTS_DIR="$BASE_RESULTS_DIR/$SCENARIO/$BAG_NAME"
        mkdir -p "$RESULTS_DIR/plots"

        echo ""
        echo "  ── Bag: $BAG_NAME ──"

        for MODE in "${MODES[@]}"; do
            for PCOUNT in "${PARTICLE_COUNTS[@]}"; do
                for ((i=1; i<=REPEATS; i++)); do

                    RESULT_NAME="${BAG_NAME}_${MODE}_${PCOUNT}p_run${i}"

                    echo "    → $MODE | ${PCOUNT}p | run${i}"

                    export BAG_FILE="$BAG"

                    rosparam set /init_particles "$PCOUNT"
                    rosparam set /max_particles  $((PCOUNT * 2))
                    rosparam set /min_particles  $((PCOUNT / 10))

                    roslaunch mcmh_localization test_algs.launch \
                        mode:=$MODE \
                        result_name:="$RESULTS_DIR/$RESULT_NAME" &

                    LAUNCH_PID=$!
                    ( sleep 100 && kill $LAUNCH_PID 2>/dev/null ) &
                    WATCHDOG_PID=$!

                    wait $LAUNCH_PID
                    kill $WATCHDOG_PID 2>/dev/null

                    if ps -p $LAUNCH_PID > /dev/null 2>&1; then
                        echo "    Processo travado, matando (PID $LAUNCH_PID)"
                        kill $LAUNCH_PID
                    fi

                    sleep 5
                done
            done
        done

        # ── offline_evaluate para este bag/cenário ────────────────────
        echo "    Calculando métricas offline: $RESULTS_DIR"
        OFFLINE_SCRIPT="$PKG_DIR/scripts/offline_evaluate.py"
        if [ -f "$OFFLINE_SCRIPT" ]; then
            RESULTS_OVERRIDE="$RESULTS_DIR" python3 "$OFFLINE_SCRIPT"
        fi

    done
done

# ── Para roscore se foi iniciado por este script ──────────────────────────
if [ -n "$ROSCORE_PID" ]; then
    kill $ROSCORE_PID 2>/dev/null
fi

# ── Gera plots para cada cenário/bag ─────────────────────────────────────
echo ""
echo "=== Gerando plots ==="

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

PLOT_SCRIPT="$PKG_DIR/scripts/plot_particle_sweep_results.py"

if [ -f "$PLOT_SCRIPT" ]; then
    for SCENARIO in "${SELECTED_SCENARIOS[@]}"; do
        SCENARIO_DIR="$BASE_RESULTS_DIR/$SCENARIO"
        if [ -d "$SCENARIO_DIR" ]; then
            echo "  Plotando cenário $SCENARIO ..."
            RESULTS_OVERRIDE="$SCENARIO_DIR" python3 "$PLOT_SCRIPT"
        fi
    done
else
    echo "Erro: script de plot não encontrado em $PLOT_SCRIPT"
fi

echo ""
echo "=== Concluído! ==="
echo "Resultados em: $BASE_RESULTS_DIR/{C,M,A}/<bag_name>/"
