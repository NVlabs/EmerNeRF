DATE=`date '+%m%d'`

scene_idx=$1
start_timestep=$2
end_timestep=$3
# reduce num_iters to 8000 for debugging
num_iters=25000

output_root="./work_dirs/$DATE"
project=feature_lifting

# use default_config.yaml for static scenes
# for novel view synthesis, change test_image_stride to 10
python train_emernerf.py \
    --config_file configs/default_dynamic.yaml \
    --output_root $output_root \
    --project $project \
    --run_name ${scene_idx}_flow \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.skip_feature_extraction=False \
    data.pixel_source.load_features=True \
    data.pixel_source.feature_model_type=dinov2_vitb14 \
    nerf.model.head.enable_feature_head=True \
    nerf.model.head.enable_learnable_pe=True \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters

# Some notes:
# 1. set `nerf.model.head.enable_learnable_pe` to False to disable learnable PE decomposition
# 2. set `nerf.model.head.enable_feature_head` to False to disable feature lifting
# 3. change `data.pixel_source.feature_model_type`per your choice