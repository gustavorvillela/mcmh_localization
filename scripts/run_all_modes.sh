#!/bin/bash

# Uso:
#   ./run_all_modes.sh                 # roda todos os bags da pasta padrão
#   ./run_all_modes.sh arquivo.bag     # roda apenas esse bag
#   ./run_all_modes.sh pasta_de_bags/  # roda todos os bags dessa pasta

MODES=("MCL" "MHMCL" "AMCL" "MHAMCL")
RESULTS_DIR="$(rospack find mcmh_localization)/results"
DEFAULT_BAG_DIR="$(rospack find mcmh_localization)/bags"
REPEATS=10   # número de repetições por cenário
mkdir -p "$RESULTS_DIR"

# Determina origem dos bags
if [ $# -eq 0 ]; then
    # Caso sem argumentos: usa pasta padrão
    BAGS=("$DEFAULT_BAG_DIR"/*.bag)
else
    BAGS=()
    for ARG in "$@"; do
        # Tenta resolver o caminho absoluto
        if [ ! -f "$ARG" ] && [ ! -d "$ARG" ]; then
            # Tenta achar no DEFAULT_BAG_DIR
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


for BAG in "${BAGS[@]}"; do
    BAG_NAME=$(basename "$BAG" .bag)
    for MODE in "${MODES[@]}"; do
        for ((i=1; i<=REPEATS; i++)); do
            echo "=== Rodando $MODE com $BAG (execução $i/$REPEATS) ==="
            export BAG_FILE="$BAG"
            RESULT_NAME="${BAG_NAME}_${MODE}_run${i}"

            roslaunch mcmh_localization test_algs.launch mode:=$MODE result_name:=$RESULT_NAME &
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
