#!/bin/bash

# Executa variações na quantidade de partículas
# Uso:
#   ./run_particle_sweep.sh
#   ./run_particle_sweep.sh L_rest.bag    # para rodar apenas esse bag

MODES=("MCL" "MHMCL" "AMCL" "MHAMCL")   # Pode ajustar conforme quiser
PARTICLE_COUNTS=(250 500 1000 2000 4000)  # valores de partículas a testar
RESULTS_DIR="$(rospack find mcmh_localization)/results"
DEFAULT_BAG_DIR="$(rospack find mcmh_localization)/bags"
REPEATS=3   # número de repetições por configuração
mkdir -p "$RESULTS_DIR"

# Determina origem dos bags
if [ $# -eq 0 ]; then
    BAGS=("$DEFAULT_BAG_DIR"/*.bag)
else
    BAGS=()
    for ARG in "$@"; do
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

# Loop principal: para cada bag, modo e número de partículas
for BAG in "${BAGS[@]}"; do
    BAG_NAME=$(basename "$BAG" .bag)
    for MODE in "${MODES[@]}"; do
        for PCOUNT in "${PARTICLE_COUNTS[@]}"; do
            for ((i=1; i<=REPEATS; i++)); do
                echo "=== Rodando $MODE com $BAG ($PCOUNT partículas, execução $i/$REPEATS) ==="
                export BAG_FILE="$BAG"
                RESULT_NAME="${BAG_NAME}_${MODE}_${PCOUNT}p_run${i}"

                roslaunch mcmh_localization test_algs.launch \
                    mode:=$MODE \
                    result_name:=$RESULT_NAME \
                    init_particles:=$PCOUNT \
                    max_particles:=$((PCOUNT * 2)) \
                    min_particles:=$((PCOUNT / 10)) &

                LAUNCH_PID=$!
                ( sleep 100 && kill $LAUNCH_PID ) & WATCHDOG_PID=$!
                wait $LAUNCH_PID
                kill $WATCHDOG_PID 2>/dev/null

                if ps -p $LAUNCH_PID > /dev/null; then
                    echo "Processo travado, matando roslaunch (PID $LAUNCH_PID)"
                    kill $LAUNCH_PID
                fi

                sleep 5
            done
        done
    done
done
