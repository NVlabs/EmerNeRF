import argparse

import numpy as np

from datasets.waymo_preprocess import WaymoProcessor

if __name__ == "__main__":
    """
    Waymo Dataset preprocessing script
    ===========================

    This script facilitates the preprocessing of the Waymo dataset

    Usage:
    ------
    python preprocess.py \
        --data_root <path_to_waymo_data> \
        --target_dir <output_directory> \
        [additional_arguments]
    
    Example:
    --------
    python preprocess.py --data_root data/waymo/raw/ --target_dir data/waymo/processed --split training --workers 3 --scene_ids 700 754 114

    Arguments:
    ----------
    --data_root (str):
        The root directory where the Waymo dataset is stored. This is a required argument.

    --split (str):
        Specifies the name of the data split. Default is set to "training".

    --target_dir (str):
        Designates the directory where the processed data will be saved. This is a mandatory argument.

    --workers (int):
        The number of processing threads. Default is set to 4.

    --scene_ids (list[int]):
        List of specific scene IDs for processing. Should be integers separated by spaces.

    --split_file (str):
        If provided, indicates the path to a file located in `data/waymo_splits` that contains the desired scene IDs.

    --start_idx (int):
        Used in conjunction with `num_scenes` to generate a list of scene IDs when neither `scene_ids` nor `split_file` are provided.

    --num_scenes (int):
        The total number of scenes to be processed.

    --process_keys (list[str]):
        Denotes the types of data components to be processed. Options include but aren't limited to "images", "lidar", "calib", "pose", etc.

    Notes:
    ------
    The logic of the script ensures that if specific scene IDs (`scene_ids`) are provided, they are prioritized. 
    If a split file (`split_file`) is indicated, it is utilized next. 
    If neither is available, the script uses the `start_idx` and `num_scenes` parameters to determine the scene IDs.
    """
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of waymo dataset"
    )
    parser.add_argument("--split", type=str, default="training", help="split name")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    # priority: scene_ids > split_file > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
        ],
    )
    args = parser.parse_args()
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    waymo_processor = WaymoProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        prefix=args.split,
        process_keys=args.process_keys,
        process_id_list=scene_ids_list,
        workers=args.workers,
    )
    if args.scene_ids is not None and args.workers == 1:
        for scene_id in args.scene_ids:
            waymo_processor.convert_one(scene_id)
    else:
        waymo_processor.convert()
