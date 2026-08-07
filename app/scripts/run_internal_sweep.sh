#!/bin/bash

# Run particle count variations
# Usage:
#   ./run_particle_sweep.sh
#   ./run_particle_sweep.sh L_rest.bag    # to run only that bag

MODES=("3MCL")   # Can adjust as desired
PARTICLE_COUNTS=(100 300 500)  # particle counts to test
STEPS_COUNTS=(10 30 50 70 80)  # particle counts to test
SCENARIOS=(M)  # C=Conservative, M=Medium, A=Aggressive
RESULTS_DIR="$(rospack find mcmh_localization)/results"
DEFAULT_BAG_DIR="$(rospack find mcmh_localization)/bags"
PARAMS_DIR="$(rospack find mcmh_localization)/params"
CLEAR=1   # Clean the results dir?
REPEATS=5   # number of repeats per configuration
MODEL="turtlebot3_${TURTLEBOT3_MODEL:-waffle}"  # TurtleBot3 model (waffle or burger)
mkdir -p "$RESULTS_DIR"
echo "Cleaning previous results..."

# Remove only generated result files (safe filter)
if $CLEAR == 1 ; then
    find "$RESULTS_DIR" -type f \( \
        -name "*.txt" -o \
        -name "*.html" -o \
        -name "*.png" \
    \) -delete
fi

scenario_profile() {
    case "$1" in
        C) echo "conservative" ;;
        M) echo "medium" ;;
        A) echo "aggressive" ;;
        *) echo "$1" ;;
    esac
}

mode_param_file() {
    local MODE="$1"
    local SCENARIO="$2"
    local PROFILE
    local MODE_LC
    local CANDIDATE

    PROFILE="$(scenario_profile "$SCENARIO")"
    MODE_LC="$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]')"
    CANDIDATE="$PARAMS_DIR/${MODE_LC}_${PROFILE}.yaml"

    if test -f "$CANDIDATE"; then
        echo "$CANDIDATE"
    elif test -f "$PARAMS_DIR/amhmcl_${PROFILE}.yaml"; then
        echo "$PARAMS_DIR/amhmcl_${PROFILE}.yaml"
    elif test -f "$PARAMS_DIR/$SCENARIO"; then
        echo "$PARAMS_DIR/$SCENARIO"
    else
        echo "Error: parameter file not found for mode '$MODE' and scenario '$SCENARIO'." >&2
        echo "Expected: $CANDIDATE" >&2
        exit 1
    fi
}

for SCENARIO in "${SCENARIOS[@]}"; do
    mkdir -p "$RESULTS_DIR/$SCENARIO/plots"
done

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
############################################
# Start roscore if it is not already running
############################################
if ! pgrep -f roscore > /dev/null; then
    echo "Starting roscore..."
    roscore &
    ROSCORE_PID=$!
fi

echo "Waiting for roscore..."
until rostopic list >/dev/null 2>&1; do
    sleep 1
done
echo "roscore is ready!"

python3 app/scripts/warmup_numba.py

# Determine source of bags
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
            echo "Warning: invalid argument ($ARG), ignored."
        fi
    done
    if [ ${#BAGS[@]} -eq 0 ]; then
        echo "Error: no valid .bag file found."
        exit 1
    fi
fi

# Main loop: for each C/M/A scenario, bag, mode and particle count
for SCENARIO in "${SCENARIOS[@]}"; do
    SCENARIO_RESULTS_DIR="$RESULTS_DIR/$SCENARIO"
    echo "=== Scenario $SCENARIO | results: $SCENARIO_RESULTS_DIR ==="

    for BAG in "${BAGS[@]}"; do
        BAG_NAME=$(basename "$BAG" .bag)
        PARAM_FILE="$(mode_param_file "$MODE" "$SCENARIO")"
        for SCOUNT in "${STEPS_COUNTS[@]}"; do
            SCENARIO_STEPS_RESULTS_DIR="$SCENARIO_RESULTS_DIR/$SCOUNT"
            for PCOUNT in "${PARTICLE_COUNTS[@]}"; do
                for ((i=1; i<=REPEATS; i++)); do
                    echo "=== Running scenario $SCENARIO | $MODE with $BAG ($PCOUNT particles, run $i/$REPEATS) ==="
                    echo "Params: $PARAM_FILE"
                    export BAG_FILE="$BAG"
                    RESULT_NAME="${BAG_NAME}_${MODE}_${PCOUNT}p_run${i}"

                    roslaunch mcmh_localization test_algs.launch \
                        mode:=$MODE \
                        result_name:=$RESULT_NAME \
                        robot_name:=$MODEL \
                        param_file:="$PARAM_FILE" \
                        results_dir:="$SCENARIO_STEPS_RESULTS_DIR" \
                        init_particles:="$PCOUNT" \
                        max_particles:="$((PCOUNT * 2))" \
                        min_particles:="$((PCOUNT / 10))" \
                        random_steps:="$SCOUNT" \
                        &

                    LAUNCH_PID=$!
                    ( sleep 100 && kill "$LAUNCH_PID" 2>/dev/null ) & WATCHDOG_PID=$!
                    wait "$LAUNCH_PID"
                    kill "$WATCHDOG_PID" 2>/dev/null

                    if ps -p "$LAUNCH_PID" > /dev/null; then
                        echo "Process hung, killing roslaunch (PID $LAUNCH_PID)"
                        kill "$LAUNCH_PID"
                    fi

                    sleep 5
                done
            done
        done
    done
done


############################################
# Stop roscore
############################################
if [ ! -z "$ROSCORE_PID" ]; then
    kill $ROSCORE_PID
fi

# Generate plots
echo "Generating plots..."

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

PLOT_SCRIPT="$(rospack find mcmh_localization)/scripts/plot_particle_sweep_results.py"
EVAL_SCRIPT="$(rospack find mcmh_localization)/scripts/offline_evaluate.py"

if [ -f "$PLOT_SCRIPT" ]; then
    python3 "$EVAL_SCRIPT"  # Generate evaluation CSVs
    python3 "$PLOT_SCRIPT"
else
    echo "Error: plot script not found!"
fi
