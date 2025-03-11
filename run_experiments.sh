#!/bin/bash

# Check if seed is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <seed>"
  exit 1
fi

SEED=$1
ENVS=("InvertedPendulum-v5" "Reacher-v5")
# ENVS=("InvertedPendulum-v5" "Walker2d-v5" "Hopper-v5" "Swimmer-v5" "Pusher-v5" "Reacher-v5")

PROJECT_NAME="sac-kan"

for ENV in "${ENVS[@]}"; do
  python3 sac_kan/sac_continuous_action.py --seed $SEED --env-id $ENV --track --wandb-project-name $PROJECT_NAME --capture-video
  python3 sac_kan/sac_continuous_action_kan.py --seed $SEED --env-id $ENV --no-cuda --track --wandb-project-name $PROJECT_NAME --capture-video
done