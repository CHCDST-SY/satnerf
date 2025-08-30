"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os


def get_opts():
    parser = argparse.ArgumentParser()

    # input paths
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/dataset/root_dir/crops_rpcs_ba_v2/JAX_068",
        help="root directory of the input dataset",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/dataset/DFC2019/Track3-RGB-crops/JAX_068",
        help="Directory where the images are located (if different than root_dir)",
    )
    parser.add_argument(
        "--ckpts_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/output/ckpts",
        help="output directory to save trained models",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/output/logs",
        help="output directory to save experiment logs",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/dataset/DFC2019/Track3-Truth",
        help="directory where the ground truth DSM is located (if available)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/Users/sy/code/py_prj/satnerf/dataset/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1",
        help="directory where cache for the current dataset is found",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="pretrained checkpoint path to load"
    )

    # other basic stuff and dataset options
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument(
        "--data",
        type=str,
        default="sat",
        choices=["sat", "blender"],
        help="type of dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sat-nerf-hash",
        choices=["nerf", "s-nerf", "sat-nerf", "sat-nerf-hash"],
        help="which NeRF to use",
    )
    parser.add_argument(
        "--gpu_id", type=int, required=False, help="GPU that will be used"
    )

    # training and network configuration
    parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size (number of input rays per iteration)",
    )
    parser.add_argument(
        "--img_downscale",
        type=float,
        default=1.0,
        help="downscale factor for the input images",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300000,
        help="number of training iterations",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=4,
        help="save checkpoints and debug files every n epochs",
    )
    parser.add_argument(
        "--fc_units",
        type=int,
        default=512,
        help="number of fully connected units in the main block of layers",
    )
    parser.add_argument(
        "--fc_layers",
        type=int,
        default=8,
        help="number of fully connected layers in the main block of layers",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=64,
        help="number of coarse scale discrete points per input ray",
    )
    parser.add_argument(
        "--n_importance",
        type=int,
        default=1,
        help="number of fine scale discrete points per input ray",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="standard deviation of noise added to sigma to regularize",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 5,
        help="maximum number of rays that can be processed at once without memory issues",
    )

    # other sat-nerf specific stuff
    parser.add_argument(
        "--sc_lambda",
        type=float,
        default=0.0,
        help="float that multiplies the solar correction auxiliary loss",
    )
    parser.add_argument(
        "--ds_lambda",
        type=float,
        default=0.0,
        help="float that multiplies the depth supervision auxiliary loss",
    )
    parser.add_argument(
        "--ds_drop",
        type=float,
        default=0.25,
        help="portion of training steps at which the depth supervision loss will be dropped",
    )
    parser.add_argument(
        "--ds_noweights",
        action="store_true",
        help="do not use reprojection errors to weight depth supervision loss",
    )
    parser.add_argument(
        "--first_beta_epoch",
        type=int,
        default=2,
        help="epoch from which transients are estimated",
    )
    parser.add_argument(
        "--t_embbeding_tau",
        type=int,
        default=4,
        help="dimension of the image-dependent embedding",
    )
    parser.add_argument(
        "--t_embbeding_vocab",
        type=int,
        default=30,
        help="Number of image-dependent embeddings, it needs to be at least the number of training images",
    )

    # hash 版
    # parser.add_argument('--bounding_box', type=list[list], default=None,
    #                     help='bounding box of the dataset')
    parser.add_argument(
        "--n_levels", type=int, default=16, help="number of levels in the hashmap"
    )
    parser.add_argument(
        "--n_features_per_level",
        type=int,
        default=2,
        help="number of features per level in the hashmap",
    )
    parser.add_argument(
        "--log2_hashmap_size",
        type=int,
        default=19,
        help="log2 of the size of the hashmap",
    )
    parser.add_argument(
        "--base_resolution",
        type=int,
        default=16,
        help="base resolution for the first hash grid level",
    )
    parser.add_argument(
        "--finest_resolution",
        type=int,
        default=512,
        help="finest resolution for the last hash grid level",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=4,
        help="Spherical harmonics degree for view/sun dir encoding",
    )

    args = parser.parse_args()

    # 当未指定 exp_name 时，使用 data+model 作为标识，避免引用未定义的 config_name
    exp_id = args.exp_name if args.exp_name is not None else f"{args.data}_{args.model}"
    args.exp_name = "{}_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id
    )
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    with open("{}/{}/opts.json".format(args.logs_dir, args.exp_name), "w") as f:
        json.dump(vars(args), f, indent=2)

    return args
