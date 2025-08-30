
## 1. Setup and Data
1. This project works with multiple conda environments, named `satnerf`, `s2p` and `ba`.

- `satnerf` is the only strictly necessary environment. It is required to train/test SatNeRF.
- `s2p` is used to additionally evaluate a satellite MVS pipeline relying on classic computer vision methods.
- `ba` is used to bundle adjust the RPCs of the DFC2019 data. 

To create the conda environments you can use the setup scripts, e.g.
```
conda init && bash -i setup_satnerf_env.sh
```

Warning: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
where `$CONDA_PREFIX` is the path to your conda or miniconda environment (e.g. `/mnt/cdisk/roger/miniconda3/envs/satnerf`).

You can download [here](https://github.com/centreborelli/satnerf/releases/tag/EarthVision2022) the training and test datasets, as well as some pretrained models.

---

## 2. Testing

Example command to generate a surface model with Sat-NeRF:
```shell
(satnerf) $ export dataset_dir=/mnt/cdisk/roger/EV2022_satnerf/dataset
(satnerf) $ export pretrained_models=/mnt/cdisk/roger/EV2022_satnerf/pretrained_models
(satnerf) $ python3 create_satnerf_dsm.py Sat-NeRF $pretrained_models/JAX_068 out_dsm_path/JAX_068 28 $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
```

Example command for novel view synthesis with Sat-NeRF:
```shell
(satnerf) $ python3 eval_satnerf.py Sat-NeRF $pretrained_models/JAX_068 out_eval_path/JAX_068 28 val $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
```
---

## 3. Training

Example command:
```shell
(satnerf) $ python3 main.py --model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth --logs_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/logs --ckpts_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/ckpts
```
---



## 4. Other functionalities


### 4.1. Dataset creation from the DFC2019 data:

The `create_satellite_dataset.py` script can be used to generate input datasets for SatNeRF from the open-source [DFC2019 data](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). The `Track3-RGB` and `Track3-Truth` folders are needed.

We encourage you to use the `bundle_adjust` package, available [here](https://github.com/centreborelli/sat-bundleadjust), to ensure your dataset employs highly accurate RPC camera models. This will also allow aggregating depth supervision to the training and consequently boost the performance of the NeRF model.
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 $dataset_dir/DFC2019 out_dataset_path/JAX_068
```

Alternatively, if you prefer not installing `bundle_adjust`, it is also possible to use the flag `--noba` to create the dataset using the original RPC camera models from the DFC2019 data.
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 $dataset_dir/DFC2019 out_dataset_path/JAX_068 --noba
```
The `--splits` flag can also be used to generate the `train.txt` and `test.txt` files.

### 4.2. Depth supervision:

The script `study_depth_supervision.py` produces an interpolated DSM with the initial depths given by the 3D keypoints output by `bundle_adjust`.

Example command:
```shell
(satnerf) $ python3 study_depth_supervision.py Sat-NeRF+DS $pretrained_models/JAX_068 out_DS_study_path/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops $dataset_dir/DFC2019/Track3-Truth
```


### 4.3. Interpolate over different sun directions:

The script `study_solar_interpolation.py` can be used to visualize images of the same AOI rendered with different solar direction vectors.

Example command:
```shell
(satnerf) $ python3 study_solar_interpolation.py Sat-NeRF $pretrained_models/JAX_068 out_solar_study_path/JAX_068 28 $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
```


### 4.4. Comparison to classic satellite MVS
We compare the DSMs learned by SatNeRF with the equivalent DSMs obtained from manually selected multiple stereo pairs, reconstructed using the [S2P](https://github.com/centreborelli/s2p) pipeline.
More details of the classic satellite MVS reconstruction process can be found [here](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Facciolo_Automatic_3D_Reconstruction_CVPR_2017_paper.html).
Use the script `eval_s2p.py` to reconstruct an AOI using this methodology.
```shell
(s2p) $ python3 eval_s2p.py JAX_068 /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/fullaoi_rpcs_ba_v1/JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/nerf_output-crops3/results --n_pairs 10
```
