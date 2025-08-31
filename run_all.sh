# run experiments

aoi_id=$1
suffix=$2
gpu_id=1
downsample_factor=1
training_iters=500000
errs="$aoi_id"_errors.txt
DFC2019_dir="/mnt/cdisk/roger/Datasets/DFC2019"
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
img_dir=$DFC2019_dir/Track3-RGB-crops/$aoi_id
out_dir="/mnt/cdisk/roger/nerf_output-crops3"
logs_dir=$out_dir/logs
ckpts_dir=$out_dir/ckpts
errs_dir=$out_dir/errs
mkdir $errs_dir
gt_dir=$DFC2019_dir/Track3-Truth


# basic NeRF
model="nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --fc_units 256"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# shadow NeRF
model="s-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# shadow NeRF + solar correction
model="s-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.05
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.05"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

#########################################################################

# satellite NeRF
model="sat-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args #2>> $errs

# satellite NeRF + solar correction
model="sat-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.1
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.1"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

#########################################################################

# satellite NeRF + solar correction (without BA)
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_raw/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_raw/"$aoi_id"_ds"$downsample_factor
model="sat-nerf"
exp_name=o_"$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.1_noBA
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.1"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# satellite NeRF + depth supervision
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
model="sat-nerf"
exp_name=o_"$aoi_id"_ds"$downsample_factor"_"$model"_DSx1000
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --ds_lambda 1000"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs


(satnerf) $ export dataset_dir=/mnt/cdisk/roger/EV2022_satnerf/dataset
(satnerf) $ export pretrained_models=/mnt/cdisk/roger/EV2022_satnerf/pretrained_models
(satnerf) $ python3 create_satnerf_dsm.py Sat-NeRF $pretrained_models/JAX_068 out_dsm_path/JAX_068 28 $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth


python3 main.py --model sat-nerf --exp_name JAX_068_ds1_sat-nerf --gpu_id 0 
--root_dir /home/sy/code/satdataset/root_dir/crops_rpcs_ba_v2/JAX_068 
--img_dir /home/sy/code/satdataset/DFC2019/Track3-RGB-crops/JAX_068 
--cache_dir /home/sy/code/satdataset/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 
--gt_dir /home/sy/code/satdataset/DFC2019/Track3-Truth 
--logs_dir /home/sy/code/satdataset/SatNeRF_output/logs 
--ckpts_dir /home/sy/code/satdataset/SatNeRF_output/ckpts


python3 main_hash.py --model sat-nerf-hash --exp_name JAX_068_ds1_sat-nerf --root_dir /root/autodl-tmp/satnerf_datasets/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /root/autodl-tmp/satnerf_datasets/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /root/autodl-tmp/satnerf_datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /root/autodl-tmp/satnerf_datasets/DFC2019/Track3-Truth --logs_dir /root/autodl-tmp/satnerf_datasets/SatNeRF_output/logs --ckpts_dir /root/autodl-tmp/satnerf_datasets/SatNeRF_output/ckpts --gpu_id 0
tensorboard --logsdir=/root/autodl-tmp/satnerf_datasets/SatNeRF_output/logs


python3 main_hash.py --model sat-nerf-hash --exp_name JAX_068_ds1_sat-nerf --root_dir /home/sy/code/satdataset/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /home/sy/code/satdataset/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /home/sy/code/satdataset/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /home/sy/code/satdataset/DFC2019/Track3-Truth --logs_dir /home/sy/code/satdataset/SatNeRF_output/logs --ckpts_dir /home/sy/code/satdataset/SatNeRF_output/ckpts --gpu_id 0
tensorboard --logdir=/home/sy/code/satdataset/SatNeRF_output/logs

