# ECTraj

This is the official GitHub repository of ["ECTraj: Enhanced Consistency Training for Multi-Agent Trajectory Prediction"](https://arxiv.org/abs/2605.08572)


> **ECTraj: Enhanced Consistency Training for Multi-Agent Trajectory Prediction**  
> Alen Mrdovic,
> Qingze (Tony) Liu,
> Danrui Li,
> Mathew Schwartz,
> Kaidong Hu,
> Sejong Yoon,
> Mubbasir Kapadia,
> Vladimir Pavlovic


<p>
<!-- <img src="assets/repo_figures/Picture1.jpg" width="1080px"/> -->

## Setup

**Step 1**: Download the code by cloning the repository:
```
git clone https://github.com/am3338/ECTraj.git
```

**Step 2**: Set up a new conda environment and install required packages:
```
conda env create -f environment.yml
conda activate ECTraj
```
**IMPORTANT:** Before running the above code, make sure to change the value in the prefix field of the environment.yml file to the location of your conda environment


## Datasets 
### Argoverse Motion Forecasting Dataset
Implement the [Argoverse 2 API](https://github.com/argoverse/av2-api) and access the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). Please see the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).

### Waymo Open Motion Dataset
Request access to the dataset on the official [Waymo website](https://waymo.com/open/) and download the dataset.

Preprocess the dataset using the preprocessing code from the [SMART GitHub repository](https://github.com/rainmaker22/SMART/blob/main/data_preprocess.py).

**WARNING: The Waymo dataset is large (~1.2TB across all three splits), so make sure in advance that you have enough space on the disk to load the entire dataset**


## Training and testing scripts

### Training Command
```sh
python train_ectraj.py --root <Path to dataset> --train_batch_size 16 --val_batch_size 4 --test_batch_size 4 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --devices "0,1,2,3" --qcnet_ckpt_path <Path to QCNet checkpoint> --num_workers 4 --num_denoiser_layers 3 --num_diffusion_steps 40 --T_max 60 --max_epochs 60 --lr 0.002 --diff_type ect-mm-fusion --ema_rate 0.0 --num_eval_samples 6 --choose_best_mode FDE --std_reg 0.3 --check_val_every_n_epoch 1 --path_pca_s_mean 'pca/<dataset>/s_mean_10.npy' --path_pca_VT_k 'pca/<dataset>/VT_k_10.npy' --path_pca_V_k 'pca/<dataset>/V_k_10.npy' --path_pca_latent_mean 'pca/<dataset>/latent_mean_10.npy' --path_pca_latent_std 'pca/<dataset>/latent_std_10.npy'

```

- `--devices`: Specifies the GPUs you want to use.
- `--qcnet_ckpt_path`: Provides the path to the QCNet checkpoint. **The checkpoint for ArgoVerse2 can be downloaded from the [QCNet repository](https://github.com/ZikangZhou/QCNet)**
- `--num_denoiser_layers`: Defines the number of layers in the denoising network.
- `--max_epochs`: Determines the total number of training epochs.
- `--lr`: Sets the learning rate.
- `--num_eval_samples`: Indicates the number of modes generated per scenario.
- `--diff_type`: Sets the type of loss during training - the type used in our final submission is `ect-mm-fusion`
- `--ema_rate`: Sets the EMA rate for the teacher network
- Ensure that you set the `<dataset>` field when loading the linear mapping matrices (last five arguments of the above script) to reflect the corresponding dataset: 'av2' for Argoverse v2 or 'waymo' for Waymo



### Validation Command
```sh
python val_ectraj.py --root <Path to dataset> --ckpt_path <Path to ECTraj checkpoint> --devices '4,5' --batch_size 8 --num_eval_samples 6 --std_reg 0.3 --path_pca_V_k 'pca/av2/V_k_10.npy'
```

## Citation
If you found our work and/or this repository helpful, please consider citing it:
```
@misc{mrdovic2026enhancingconsistencymodelsmultiagent,
      title={Enhancing Consistency Models for Multi-Agent Trajectory Prediction}, 
      author={Alen Mrdovic and Qingze and Liu and Danrui Li and Mathew Schwartz and Kaidong Hu and Sejong Yoon and Mubbasir Kapadia and Vladimir Pavlovic},
      year={2026},
      eprint={2605.08572},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.08572}, 
}
```

We thank the authors of the following repositories for open-sourcing their code:
- [Query-Centric Trajectory Prediction](https://github.com/ZikangZhou/QCNet).
- [Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](https://github.com/YixiaoWang7/OptTrajDiff)
- [SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction](https://github.com/rainmaker22/SMART)

Please also consider citing their work:
```
@inproceedings{zhou2023query,
  title={Query-Centric Trajectory Prediction},
  author={Zhou, Zikang and Wang, Jianping and Li, Yung-Hui and Huang, Yu-Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

@inproceedings{wang2024optimizing,
  title={Optimizing diffusion models for joint trajectory prediction and controllable generation},
  author={Wang, Yixiao and Tang, Chen and Sun, Lingfeng and Rossi, Simone and Xie, Yichen and Peng, Chensheng and Hannagan, Thomas and Sabatini, Stefano and Poerio, Nicola and Tomizuka, Masayoshi and others},
  booktitle={European conference on computer vision},
  pages={324--341},
  year={2024},
  organization={Springer}
}

@article{wu2024smart,
  title={Smart: Scalable multi-agent real-time motion generation via next-token prediction},
  author={Wu, Wei and Feng, Xiaoxin and Gao, Ziyan and Kan, Yuheng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={114048--114071},
  year={2024}
}
```
