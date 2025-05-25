## Installation

```bash
conda env create --file environment.yml
conda activate genfusion
cd Reconstruction
CC=gcc-9 CXX=g++-9 pip install submodules/simple-knn
CC=gcc-9 CXX=g++-9 pip install submodules/diff-surfel-rasterization
```
## Generation Model Training可以跳过

The generation model is finetuned from [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter).

Step 1. Download the DL3DV Renderings dataset

Step 2. Download pretrained models via Hugging Face, and put the model.ckpt with the required resolution in checkpoints/dynamicrafter_512_v1/model.ckpt.

Step 3. Run the following command to start training on a resolution of 512x320.

```bash
cd ./GenerationModel
sh configs/training_video_v1.0/run_interp.sh
```
Step 4. Finetuned the model on a higher resolution of 960x512.
```bash
sh configs/training_960_v1.0/run_interp.sh
```
If you want to use the model for generation inference, you can download our pre-trained model from [here](https://huggingface.co/Sibo2rr/GenFusion-GenerationModel).

## Reconstruction

diffusion_ckpt需要放在Reconstruction下

If your skip the generation model training, you can download our pre-trained model from [here](https://huggingface.co/Sibo2rr/GenFusion-GenerationModel) and put it in the `./diffusion_ckpt` folder

### Masked 3D Reconstruction

data需要放在Reconstruction下，同时把那些以hash命名的文件夹直接放在data下

下载 `https://cdn-lfs.hf-mirror.com/repos/a4/00/a400d3407dcb6ba1378f2783695f545ab7ba55b1b8bbea86f0eb7ab8793c3a98/9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27open_clip_pytorch_model.bin%3B+filename%3D%22open_clip_pytorch_model.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1748188586&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0ODE4ODU4Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy9hNC8wMC9hNDAwZDM0MDdkY2I2YmExMzc4ZjI3ODM2OTVmNTQ1YWI3YmE1NWIxYjhiYmVhODZmMGViN2FiODc5M2MzYTk4LzlhNzhlZjhlOGM3M2ZkMGRmNjIxNjgyZTdhOGU4ZWIzNmM2OTE2Y2IzYzE2YjI5MWEwODJlY2Q1MmFiNzljYzQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=av6SQcQgSVeBJJ1uc5lpYCLmhFsfEEpnqw0MAKgn50zqAZoGntnofF62pwoH228z0XE1MUrN3TdZfiJTFM6l49Edipfk2TjBqSBH0ui5gAJMHiuRldkF3anOXctOix6dwgdTiD3VulUJKhraJM8nXZ8ZaF1BwttepacTn4cLISBDTVO24Jbc3XbhezeHQvcNEWFmTEbRsZKy%7ES-wejzN13RDC77C7VhnrOJ62xus00dKZ2eqtOY0Y-a7ZTbRVA9aWEyWNVKM625VMo0eIosvw1fPbBm2Zkkmy9so%7EEY-yjHqV0YjfdOywal72ll2BowGFMZLtB58D2h8pzblFhB0yg__&Key-Pair-Id=K3RPWS32NSSJCE`放在/home/caoyuanping/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c下

train.py最前面加：
```
import os
os.environ["HF_ENDPOINT"]= "https://hf-mirror.com"
```

模型命名为epoch=59-step=34000.ckpt

第一次运行的可能会报错没有output，把相应的注释掉就行或者自己创建一个试一试。

Step 1. The testing scenes in our paper is selected from DL3DV Benchmarkand DL3DV dataset. Download scenes from [Huggingface](https://huggingface.co/datasets/Inception3D/GenFusion_DL3DV_24Benchmark) and put them in the `./data` folder.

Step 2. To get the quantitative results in our paper, you can run the following command.

```bash
cd ./Reconstruction
python genfusion_scripts/batch_ours.py [gpu_ids]
```

To run a masked 3D reconstruction on your own scene, you can use the following command.

```bash
cd ./Reconstruction
python train.py --data_dir /home/caoyuanping/workspace/genfusion/Reconstruction/data/newMeetingRoom -m /home/caoyuanping/workspace/genfusion/Reconstruction/output_dir --iterations 7_000 --test_iterations 7_000 --diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt --diffusion_config ./generation_infer.yaml --num_frames 16 --outpaint_type crop --start_diffusion_iter 3000 --sparse_view 0 --downsample_factor 2 --diffusion_resize_width 960 --diffusion_resize_height 512 --diffusion_crop_width 960 --diffusion_crop_height 512 --patch_size 360 640 --opacity_reset_interval 9000 --lambda_dist 0.0 --lambda_reg 0.5 --lambda_dssim 0.8 --densify_from_iter 1000 --unconditional_guidance_scale 3.2 --ip 0.0.0.0 --port 7861 --repair
```

### Sparse view Reconstruction

Step 1. Download the Mip-NeRF data and train/test split from [ReconFusion](https://drive.google.com/drive/folders/10oT2_OQ9Sjh5wlfJQoGx2y7ZKYwpgNg5)

Step 2. Run the following command to start training.

```bash
cd ./Reconstruction
python genfusion_scripts/batch_sparse.py [gpu_ids]
```

To run a sparse view reconstruction on your own scene, a `train_test_split_{sparse_view}.json` file is required in the following format:

```json
{
  "test_ids": [
    0, 8, 12, ...
  ],
  "train_ids": [
    1, 2, 3, ...
  ]
}
```

Then run the following command:

```bash
python train.py \
--data_dir [data_dir] 
-m [output_dir] \
--iterations 7000 \
--test_iterations 7_000 \
--diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt \
--diffusion_config ./generation_infer.yaml \
--num_frames 16 \
--outpaint_type sparse \
--start_diffusion_iter 1000 \
--sparse_view [num_of_sparse_views] \
--diffusion_resize_width 960 \
--diffusion_resize_height 512 \
--diffusion_crop_width 960 \
--diffusion_crop_height 512 \
--mono_depth \
--repair \
--densify_from_iter 100 \
--diffusion_until 7000 \
--diffusion_every 1000 \
--densify_until_iter 5000 \
--densification_interval 500 \
--opacity_reset_interval 3100 \
--lambda_dist 10.0 \
--lambda_dssim 0.5 \
--lambda_reg 1.0 \
--unconditional_guidance_scale 3.2
```

### Scene Completion
We provide two options to reconstruct the unseen area of the scene.

- Option 1: Pre-defined camera path using [RemoteViewer](https://github.com/hwanhuh/2D-GS-Viser-Viewer), save the trajectory as a json file. Use counter scene in Mip-NeRF360 as an example:
```bash

python train.py \
    --data_dir [data_dir] \
    -m output_ours/counter_completion \
    --test_iterations 7_000 \
    --diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt \
    --diffusion_config ./generation_infer.yaml \
    --num_frames 16 \
    --outpaint_type rotation \
    --add_indices 7 15 \
    --depth_loss \
    --iterations 26000 \
    --diffusion_resize_width 960 \
    --diffusion_resize_height 512 \
    --diffusion_crop_width 960 \
    --diffusion_crop_height 512 \
    --repair \
    --port 6678 \
    --densify_from_iter 500 \
    --densify_until_iter 12000 \
    --diffusion_until 30000 \
    --start_diffusion_iter 5000 \
    --diffusion_every 4000 \
    --opacity_reset_interval 15000 \
    --unconditional_guidance_scale 2.2 \
    --start_dist_iter 3000 \
    --camera_path_file [trajectory file]
```

- Option 2: Use path paramters in the script to determine the shape of sampling path.

```bash
python train.py \
    --data_dir [data_dir] \
    -m [output_dir] \
    --test_iterations 7_000 \
    --diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt \
    --diffusion_config ./generation_infer.yaml \
    --num_frames 16 \
    --outpaint_type rotation \
    --add_indices 7 15 \
    --depth_loss \
    --iterations 26000 \
    --diffusion_resize_width 960 \
    --diffusion_resize_height 512 \
    --diffusion_crop_width 960 \
    --diffusion_crop_height 512 \
    --port 6691 \
    --densify_from_iter 500 \
    --densify_until_iter 12000 \
    --diffusion_until 30000 \
    --start_diffusion_iter 5000 \
    --diffusion_every 4000 \
    --opacity_reset_interval 15000 \
    --repair \
    # define the following parameters for your own scene
    --path_scale 1.0 \ 
    --rotation_angle 90 \
    --position_z_offset 0.5 \
    --distance -0.2 \
    --unconditional_guidance_scale 2.2 \
    --start_dist_iter 3000
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{Wu2025GenFusion,
    author = {Sibo Wu and Congrong Xu and Binbin Huang and Geiger Andreas and Anpei Chen},
    title = {GenFusion: Closing the Loop between Reconstruction and Generation via Videos},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025}
}
```


## Acknowledgements

This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter). The dataset processing is based on [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main/gsplat). We thank all the authors for their great repos. 


Special thanks to [LiangrunDa](https://github.com/LiangrunDa) for his help on the project website and part of the code. 

