### Features

-  VAE Model 



**Table of Contents**

[TOCM]

[TOC]



##Main Script

The entry point to the program is *main.py* in the parent directory. The script is executed with the following syntax 
```shell
python main.py --config <path/to/config/file> --<function to run> --<function arguments>
```
Note : The configuration file is required for script execution.
#### Arguments

**VAE**

|  Function | Description   | Function Arguments   |
| ----------- | ------------ | ------------ |
|  `train_vae` |  Starts a VAE training run | `--checkpoint` : ( *Optional* ) Path to  checkpoint file to start training run from.|
| `sample_vae`  | Generate data from encoded VAE latents| `--checkpoint` : Path to  checkpoint file with trained VAE . <br>`--name` : ( *Optional*  ) output file name to save as. <br>
| `reconstruct_vae`  | Generate (encode and decode) a sample from a trained VAE  |`--checkpoint` : Path to  checkpoint file with trained VAE . <br>`--name` : ( *Optional*  ) output file name to save as. <br> |
| `encode`  | Encode data and generate latents  |`--vae_checkpoint` : Path to  checkpoint file with trained VAE . <br>`--input_dir` : Directory path for input videos to be encoded. <br>`--output_dir` : Directory path to write encoded frames for Diffusion Transformer training. <br>|


**Diffusion**

|  Function | Description   | Function Arguments   |
| ----------- | ------------ | ------------ |
|  `train_diffusion` |  Starts a VAE training run | `--checkpoint` : ( *Optional* ) Path to  checkpoint file to start training run from.|
| `sample_diffusion`  | Generate (encode and decode) a sample from a trained VAE |`--vae_checkpoint` : Path to  checkpoint file with trained VAE . <br>`--diffusion_checkpoint` : Path to  checkpoint file with trained Diffusion Model . <br>`--data_dir` : Directory path with video latents. <br>`--name` : ( *Optional*  ) output file name to save as. <br>|

**Misc**

|  Function | Description   | Function Arguments   |
| ----------- | ------------ | ------------ |
|  `plot_loss` |  Plot training run loss  | `--type` : The loss to plot, can either be `vae` or `dt`
| `make_video`  | Make video from generated frames |`--data_dir` : The folder contaning image frames. <br>`--name` : ( *Optional*  ) output file name to save as. <br>|
## Configuration

The configuration file are in *JSON*  format. A single configuration file can be used for all the functions in *main.py*.


```json
{
    "seed":  <Integer>  Seed to use for stochastic operation
    "lvm": {
        "n_latent": <Integer>  Number of latents to be used for encoded frames
    },
    "transcode": {
        "bs": <Integer>  Training Batch Size,
        "target_size":<Array> [Width, Height ] Size of generated frames
    },
    "vae": {
        "size_multiplier": <Integer> Size multiplier for VAE architecture
        "sample": {
            "n_sample": <Integer>  Number of samples to be generated.
        },
        "reconstruct": {
            "n_sample": <Integer>  Number of samples to be generated
            "video_file":<Path>  Input Video File
            "generation_path":  <Path>  Directory for reconstucted samples to be saved
        },
        "train": {
            "lr": <Float>  Learning Rate,
            "data_dir_train": <Path>  Training Data Directory,
            "data_dir_val":  <Path>  Validation Data Directory,
            "bs": <Integer> Batch Size,
            "metrics_path": <Path>  Log File Path,
            "clip_norm": <Float> Clip Norm,
            "kl_alpha": <Float> KL Divergence Regularisaiton Term (Beta)
        }
    },
    "dt": {
        "n_layers": <Integer> Number of Layers in Diffusion Transformer,
        "d_l": <Integer> Tunable Hyperparameter,
        "d_mlp": <Integer> Tunable Hyperparameter,
        "n_q": <Integer> Tunable Hyperparameter,
        "d_qk": <Integer> Tunable Hyperparameter,
        "d_dv": <Integer> Tunable Hyperparameter,
        "l_x": <Integer> Tunable Hyperparameter,
        "l_y": <Integer> Tunable Hyperparameter,
        "sample": {
            "n_sample": <Integer> Number of Diffusion Samples to be generated,
            "n_steps": <Integer> Diffusion Steps,
            "generation_path": <Path>  Directory for samples to be saved
        },
        "train": {
            "ckpt_dir": <Path>  Directory to save checkpoint names for diffusion
            "lr": <Float> Learning Rate 
            "ckpt_interval": <Integer>  Checkpoint Save Interval,
            "data_dir_train": <Path>  Training Data Directory,
            "data_dir_val": <Path>  Validation Data Directory",
            "bs": <Integer> Batch Size
            "metrics_path": <Path> Log File Path
            "vae_checkpoint": <Path>  Validation Data Directory,
            "clip_norm":  <Float> Clip Norm
        }
    },
    "checkpoints": {
        "ckpt_dir": <Path>  Directory to save checkpoint names,
        "ckpt_name": <String>  Checkpoint Name Identifier,
        "ckpt_interval": <Integer>  Checkpoint Save Interval,
    }
}
```


###End