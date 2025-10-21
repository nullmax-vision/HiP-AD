#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True

CONFIG_NAME=hipad_b2d_stage2

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG=/opt/data/private/project/HiP-AD/projects/configs/$CONFIG_NAME.py+\
/opt/data/private/project/HiP-AD/work_dirs/$CONFIG_NAME/latest.pth

PLANNER_TYPE=traj
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220

SAVE_PATH=evaluation/$CONFIG_NAME
BASE_CHECKPOINT_ENDPOINT=evaluation/$CONFIG_NAME/$CONFIG_NAME

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi

echo -e "**************Please Manually adjust GPU or TASK_ID **************"
GPU_RANK_LIST=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)
TASK_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

echo -e "TASK_LIST: $TASK_LIST"
echo -e "GPU_RANK_LIST: $GPU_RANK_LIST"
echo -e "\033[36m***********************************************************************************\033[0m"

length=${#TASK_LIST[@]}

start_task() {
    local i=$1
    PORT=$((BASE_PORT + i * 200))
    TM_PORT=$((BASE_TM_PORT + i * 200))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    echo -e "\033[32m Starting task $i on rank $GPU_RANK \033[0m"

    # Kill Python and Carla processes for this task
    pkill -f "python.*${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py.*--port=${PORT}"
    sleep 3

    # Start the task
    bash -e bench2drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.log &
    echo -e "bash bench2drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK"
    echo -e "\033[36m***********************************************************************************\033[0m"
}

for ((i=0; i<$length; i++ )); do
    start_task $i
    sleep 5
done

# Monitor and restart loop
check_and_restart_task() {
    local i=$1
    local log_file=$2
    local timeout=300  # 5 minutes, adjust as needed

    if [ -f "$log_file" ]; then
        # Check for engine crash
        if grep -q "Engine crash handling finished" "$log_file"; then
            echo -e "\033[31m Task $i has crashed (Engine crash found). Restarting... \033[0m"
            start_task $i
            sleep 5
            return
        fi

        # Check for timeout
        last_modified=$(stat -c %Y "$log_file")
        current_time=$(date +%s)
        time_diff=$((current_time - last_modified))

        if [ $time_diff -gt $timeout ]; then
            echo -e "\033[33m Task $i appears to be stuck (no log updates for ${timeout}s). Restarting... \033[0m"
            start_task $i
            sleep 5
        fi
    fi
}

while true; do
    for ((i=0; i<$length; i++ )); do
        log_file="${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.log"
        check_and_restart_task $i $log_file
    done
    sleep 60  # Check every minute
done

wait