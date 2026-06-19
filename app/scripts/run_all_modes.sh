#!/bin/bash

# Usage:
#   ./run_all_modes.sh                 # run all bags in the default folder
#   ./run_all_modes.sh file.bag       # run only that bag
#   ./run_all_modes.sh bags_folder/   # run all bags in that folder

MODES=("MCL" "MHMCL" "MHAMCL" "AMCL" "AMHMCL" "AMHAMCL")
#MODES=("MCL" "MHMCL" "AMHMCL")
#MODES=("AMCL" "MHAMCL" "AMHAMCL")

RESULTS_DIR="$(rospack find mcmh_localization)/results"
DEFAULT_BAG_DIR="$(rospack find mcmh_localization)/bags"
REPEATS=10   # number of repeats per scenario
mkdir -p "$RESULTS_DIR"

echo -e "\nModes:  (${MODES[*]})\nResults dir: $RESULTS_DIR\nRepeats per mode: $REPEATS\n" 

# Determine source of bags
if [ $# -eq 0 ]; then
    # If no arguments: use default folder
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
            echo "Warning: invalid argument ($ARG), ignored."
        fi
    done

    if [ ${#BAGS[@]} -eq 0 ]; then
        echo "Error: no valid .bag file found."
        exit 1
    fi
fi


for BAG in "${BAGS[@]}"; do
    BAG_NAME=$(basename "$BAG" .bag)
    for MODE in "${MODES[@]}"; do
        for ((i=1; i<=REPEATS; i++)); do
            echo -e "\n\n=== Running $MODE with $BAG (run $i/$REPEATS) ===\n\n"
            export BAG_FILE="$BAG"
            RESULT_NAME="${BAG_NAME}_${MODE}_run${i}"

            roslaunch mcmh_localization test_algs.launch mode:=$MODE result_name:=$RESULT_NAME &
            LAUNCH_PID=$!

            ( sleep 100 && kill $LAUNCH_PID ) & WATCHDOG_PID=$!
            wait $LAUNCH_PID
            rosnode kill -a
            sleep 2
            kill $WATCHDOG_PID 2>/dev/null

            if ps -p $LAUNCH_PID > /dev/null; then
                echo "Process hung, killing roslaunch (PID $LAUNCH_PID)"
                rosnode kill -a
                sleep 2
                kill $LAUNCH_PID
            fi

            sleep 5
        done
    done
done
