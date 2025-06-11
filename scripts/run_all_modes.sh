#!/bin/bash

BAG_DIR="$(rospack find mcmh_localization)/bags"
MODES=("MCL" "MHMCL" "AMCL" "MHAMCL")
RESULTS_DIR="$(rospack find mcmh_localization)/results"
mkdir -p "$RESULTS_DIR"

for BAG in "$BAG_DIR"/*.bag; do
    BAG_NAME=$(basename "$BAG" .bag)
    for MODE in "${MODES[@]}"; do
        echo "=== Rodando $MODE com $BAG ==="
        export BAG_FILE="$BAG"
        RESULT_NAME="${BAG_NAME}_${MODE}"

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
