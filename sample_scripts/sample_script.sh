# !/bin/bash

DATE=`date '+%m%d'`

scene_idx=$1
start_timestep=$2
end_timestep=$3
# reduce num_iters to 8000 for debugging
num_iters=25000

output_root="./work_dirs/$DATE"
project=scene_reconstruction

# use default_config.yaml for static scenes
# for novel view synthesis, change test_image_stride to 10
# with flow field
python train_emernerf.py \
    --config_file configs/default_flow.yaml \
    --output_root $output_root \
    --project $project \
    --run_name ${scene_idx}_flow \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters


# without flow field
python train_emernerf.py \
    --config_file configs/default_dynamic.yaml \
    --output_root $output_root \
    --project $project \
    --run_name ${scene_idx}_wo_flow \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters
