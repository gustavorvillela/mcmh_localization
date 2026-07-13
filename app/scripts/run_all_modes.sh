#!/bin/bash

# Usage:
#   ./run_all_modes.sh                 # run all bags in the default folder
#   ./run_all_modes.sh file.bag       # run only that bag
#   ./run_all_modes.sh bags_folder/   # run all bags in that folder

MODES=("MCL" "MHMCL" "MHAMCL" "AMCL" "AMHMCL" "AMHAMCL")
#MODES=("MCL" "MHMCL" "AMHMCL")
#MODES=("AMCL" "MHAMCL" "AMHAMCL")
SCENARIOS=("C" "M" "A")

RESULTS_DIR="$(rospack find mcmh_localization)/results"
DEFAULT_BAG_DIR="$(rospack find mcmh_localization)/bags"
PARAMS_DIR="$(rospack find mcmh_localization)/params"
REPEATS=30   # number of repeats per scenario
mkdir -p "$RESULTS_DIR"

scenario_param_file() {
    case "$1" in
        C) echo "$PARAMS_DIR/amhmcl_conservative.yaml" ;;
        M) echo "$PARAMS_DIR/amhmcl_medium.yaml" ;;
        A) echo "$PARAMS_DIR/amhmcl_aggressive.yaml" ;;
        *)  if test -f "$PARAMS_DIR/$1"; then
                echo "$PARAMS_DIR/$1"
            else
                echo "Error: scenario '$1' is not on '$PARAMS_DIR'. Use C, M, A or a valid file." >&2
                exit 1
            fi ;;
    esac
}

for SCENARIO in "${SCENARIOS[@]}"; do
    mkdir -p "$RESULTS_DIR/$SCENARIO/plots"
done

echo -e "\nScenarios: (${SCENARIOS[*]})\nModes:  (${MODES[*]})\nResults dir: $RESULTS_DIR\nRepeats per mode: $REPEATS\n" 

python3 app/scripts/warmup_numba.py

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


for SCENARIO in "${SCENARIOS[@]}"; do
    PARAM_FILE="$(scenario_param_file "$SCENARIO")"
    SCENARIO_RESULTS_DIR="$RESULTS_DIR/$SCENARIO"

    if [ ! -f "$PARAM_FILE" ]; then
        echo "Error: parameter file not found: $PARAM_FILE"
        exit 1
    fi

    for BAG in "${BAGS[@]}"; do
        BAG_NAME=$(basename "$BAG" .bag)
        for MODE in "${MODES[@]}"; do
            for ((i=1; i<=REPEATS; i++)); do
                echo -e "\n\n=== Running scenario $SCENARIO | $MODE with $BAG (run $i/$REPEATS) ===\n\n"
                export BAG_FILE="$BAG"
                RESULT_NAME="${BAG_NAME}_${MODE}_run${i}"

                roslaunch mcmh_localization test_algs.launch \
                    mode:=$MODE \
                    result_name:=$RESULT_NAME \
                    param_file:="$PARAM_FILE" \
                    results_dir:="$SCENARIO_RESULTS_DIR" \
                    &
                LAUNCH_PID=$!

                ( sleep 100 && kill "$LAUNCH_PID" 2>/dev/null ) & WATCHDOG_PID=$!
                wait "$LAUNCH_PID"
                rosnode kill -a
                sleep 2
                kill "$WATCHDOG_PID" 2>/dev/null

                if ps -p "$LAUNCH_PID" > /dev/null; then
                    echo "Process hung, killing roslaunch (PID $LAUNCH_PID)"
                    rosnode kill -a
                    sleep 2
                    kill "$LAUNCH_PID"
                fi

                sleep 5
            done
        done
    done
done
