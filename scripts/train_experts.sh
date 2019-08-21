#!/usr/bin/env bash

# This script trains experts for AIRL and GAIL benchmark scripts.
# When training is finished, it reports the mean episode reward of each
# expert and builds a zip file of expert models that can be used for
# `experiments/gail_benchmark.sh`.

ENVS+="acrobot cartpole mountain_car "
ENVS+="reacher half_cheetah hopper ant humanoid swimmer walker "
ENVS+="two_d_maze custom_ant disabled_ant "

SEEDS="0 1 2"

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
OUTPUT_DIR="output/train_experts/${TIMESTAMP}"
RESULTS_FILE="results.txt"

echo "Writing logs in ${OUTPUT_DIR}"
# Train experts.
parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
  python -m airl.scripts.expert_demos \
  with \
  {env} \
  seed={seed} \
  log_root=${OUTPUT_DIR} \
  ::: env ${ENVS} ::: seed ${SEEDS}

pushd $OUTPUT_DIR
shopt -s failglob  # Catch obvious errors by reporting glob match fails.

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout | xargs tail -n 50 | grep -E '(==|ep_reward_mean)' | tee ${RESULTS_FILE}
popd
