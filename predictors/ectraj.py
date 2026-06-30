# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE, BestMinJADE, BestMinJFDE, minJADE, minJFDE, meanJADE, meanJFDE, minJLDE, meanJLDE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from metrics import meanJFDEG, minJFDEG
from metrics import meanVelRate, maxVelRate, targetVelminError, targetVelmeanError, KinematicFeasibleRate
from metrics import CR
from predictors import QCNet
from modules import JointCM
import numpy as np
from time import time
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from av2.datasets.motion_forecasting import scenario_serialization
from visualization import *
from av2.map.map_api import ArgoverseStaticMap

import os


class DiffNet(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super(DiffNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.output_head = args.output_head
        self.num_historical_steps = args.num_historical_steps
        self.num_future_steps = args.num_future_steps
        self.num_modes = args.num_modes
        self.num_recurrent_steps = args.num_recurrent_steps
        self.num_freq_bands = args.num_freq_bands
        self.num_map_layers = args.num_map_layers
        self.num_agent_layers = args.num_agent_layers
        self.num_dec_layers = args.num_dec_layers
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.dropout = args.dropout
        self.pl2pl_radius = args.pl2pl_radius
        self.time_span = args.time_span
        self.pl2a_radius = args.pl2a_radius
        self.a2a_radius = args.a2a_radius
        self.num_t2m_steps = args.num_t2m_steps
        self.pl2m_radius = args.pl2m_radius
        self.a2m_radius = args.a2m_radius
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T_max = args.T_max
        self.submission_dir = args.submission_dir
        self.submission_file_name = args.submission_file_name
        self.diff_type = args.diff_type
        self.sampling = args.sampling
        self.sampling_stride = args.sampling_stride
        self.num_diffusion_steps = args.num_diffusion_steps
        self.num_eval_samples = args.num_eval_samples
        self.eval_mode_error_2 = args.eval_mode_error_2
        self.choose_best_mode = args.choose_best_mode
        self.train_agent = args.train_agent
        self.path_pca_s_mean = args.path_pca_s_mean
        self.path_pca_VT_k = args.path_pca_VT_k
        self.path_pca_latent_mean = args.path_pca_latent_mean
        self.path_pca_latent_std = args.path_pca_latent_std
        self.s_mean = None
        self.VT_k = None
        self.latent_mean = None
        self.latent_std = None
        self.m_dim = args.m_dim

        self.check_param()

        self.qcnet = QCNet.load_from_checkpoint(checkpoint_path=args.qcnet_ckpt_path)
        self.qcnet.freeze()

        self.linear = nn.Linear(10, 2)

        self.joint_diffusion = JointCM(args=args)

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * args.output_dim + ['von_mises'] * args.output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(
            component_distribution=['laplace'] * args.output_dim + ['von_mises'] * args.output_head,
            reduction='none')

        self.Brier = Brier(max_guesses=6)

        self.minAHE = minAHE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)
        self.MR_c = MR(max_guesses=6)

        self.CR = CR(max_guesses=6)
        self.CR_c = CR(max_guesses=6)

        self.test_predictions = dict()
        self.BestMinJADE = BestMinJADE(max_guesses=6)
        self.BestMinJFDE = BestMinJFDE(max_guesses=6)

        self.minJADE = minJADE(max_guesses=6)
        self.minJFDE = minJFDE(max_guesses=6)

        self.minJADE_diff = minJADE(max_guesses=6)
        self.minJFDE_diff = minJFDE(max_guesses=6)

        self.minJADE_diff_c = minJADE(max_guesses=6)
        self.minJFDE_diff_c = minJFDE(max_guesses=6)

        self.minJADE_diff_QCNet = minJADE(max_guesses=6)
        self.minJFDE_diff_QCNet = minJFDE(max_guesses=6)

        self.minADE_diff = minADE(max_guesses=6)
        self.minFDE_diff = minFDE(max_guesses=6)

        self.minADE_diff_c = minADE(max_guesses=6)
        self.minFDE_diff_c = minFDE(max_guesses=6)

        self.minJLDE = minJLDE()
        self.meanJLDE = meanJLDE()

        self.minJFDEG = minJFDEG()
        self.meanJFDEG = meanJFDEG()

        self.meanJADE_diff_c = meanJADE(max_guesses=6)
        self.meanJFDE_diff_c = meanJFDE(max_guesses=6)

        self.minJADE_diff_gmm = minJADE(max_guesses=6)
        self.minJFDE_diff_gmm = minJFDE(max_guesses=6)

        self.meanVelRate = meanVelRate()
        self.maxVelRate = maxVelRate()
        self.KinematicFeasibleRate = KinematicFeasibleRate()

        self.KinematicConfortRate = KinematicFeasibleRate()

        self.targetVelminError = targetVelminError()
        self.targetVelmeanError = targetVelmeanError()

        self.num_all_agents = 0
        self.M_dis = []
        self.order_ac = []


    def add_extra_param(self, args):
        self.std_reg = args.std_reg
        self.path_pca_V_k = args.path_pca_V_k
        self.V_k = None

        self.num_sampling_steps = args.num_sampling_steps


    def check_param(self):
        if self.sampling == 'ddpm':
            self.sampling_stride = 1
        elif self.sampling == 'ddim':
            self.sampling_stride = int(self.sampling_stride)
            if self.sampling_stride > self.num_diffusion_steps - 1:
                print('ddim stride > diffusion steps.')
                exit()
            scale = self.num_diffusion_steps / self.sampling_stride
            if abs(scale - int(scale)) > 0.00001:
                print('mod(diffusion steps, ddim stride) != 0')
                exit()

    def forward(self, data: HeteroData):
        scene_enc = self.qcnet.encoder(data)
        x = torch.ones(32, 10).to(scene_enc['x_a'].device)
        return self.linear(x)

    def normalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0)) / (std.unsqueeze(0) + 0.1)
            if mean.dim() == 2:
                return (original_data - mean) / (std + 0.1)
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0).unsqueeze(0)) / (std.unsqueeze(0).unsqueeze(0) + 0.1)
            if mean.dim() == 2:
                return (original_data - mean.unsqueeze(1)) / (std.unsqueeze(1) + 0.1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')

    def unnormalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return original_data * (std.unsqueeze(0) + 0.1) + mean.unsqueeze(0)
            if mean.dim() == 2:
                return original_data * (std + 0.1) + mean
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return original_data * (std.unsqueeze(0).unsqueeze(0) + 0.1) + mean.unsqueeze(0).unsqueeze(0)
            if mean.dim() == 2:
                return original_data * (std.unsqueeze(1) + 0.1) + mean.unsqueeze(1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')

    def training_step(self,
                      data,
                      batch_idx):  # batch_

        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)


        eval_mask = (data['agent']['category'] >= 2) & (reg_mask[:, -1] == True) & (reg_mask[:, 0] == True)
        mask = eval_mask
        gt = gt[mask][..., :self.output_dim]
        reg_mask = reg_mask[mask]
        num_agent = gt.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(gt.shape[1] - 1):
                if reg_mask[i, j] == True and reg_mask[i, j + 1] == False:
                    start.append(j)
                elif reg_mask[i, j] == False and reg_mask[i, j + 1] == True:
                    end.append(j + 1)
            reg_start_list.append(start)
            reg_end_list.append(end)

        for i in range(num_agent):
            count = 0
            for j in range(gt.shape[1] - 1):
                if reg_mask[i, j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt[i, start_id]
                    end_pt = gt[i, end_id]
                    gt[i, j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j - start_id)
                    if j == end_id - 1:
                        count += 1

        flat_gt = gt.reshape(gt.size(0), -1)
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0, 1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)

        target_mode = torch.matmul(flat_gt - self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)

        marginal_trajs = traj_refine[eval_mask, :, :, :2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0), self.num_modes, -1)

        marginal_mode = torch.matmul((marginal_trajs - self.s_mean.unsqueeze(1)).permute(1, 0, 2),
                                     self.VT_k.unsqueeze(0).repeat(self.num_modes, 1, 1))
        marginal_mode = marginal_mode.permute(1, 0, 2)

        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)

        marg_mean = marginal_mode.mean(dim=1)
        marg_std = marginal_mode.std(dim=1) + self.std_reg

        mean = marg_mean
        std = marg_std

        loss = self.joint_diffusion.get_loss(target_mode, data=data, scene_enc=scene_enc, gt_future=gt, mean=mean, std=std, num_modes=6,
                                             mm=marginal_mode, mmscore=pi.exp()[eval_mask], eval_mask=eval_mask, epoch=self.current_epoch + 1, max_epochs=self.trainer.max_epochs)

        print("LOSS: ", loss)

        self.log('train_loss', loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        if self.diff_type in ['cm', 'ict', 'ict-mm', 'ect-mm', 'ict-mm-fusion', 'ect-mm-fusion']:
            self.joint_diffusion.update_teacher()

        if print_flag:
            print(batch_idx, loss)

        return loss

    def validation_step(self,
                             data,
                             batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True

        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        traj_his = data['agent']['position'][:, :self.num_historical_steps,:]
        gt_his = torch.cat([traj_his[..., :self.output_dim], traj_his[..., -1:]], dim=-1)

        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0, 1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)

        mask = (data['agent']['category'] >= 2) & (reg_mask[:, -1] == True) & (reg_mask[:, 0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        gt_n[0, :, :] = (gt_n[0, :, :] - gt_n[0, 0:1, :]) / 4 * 3 + gt_n[0, 0:1, :]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(gt.shape[1] - 1):
                if reg_mask_n[i, j] == True and reg_mask_n[i, j + 1] == False:
                    start.append(j)
                elif reg_mask_n[i, j] == False and reg_mask_n[i, j + 1] == True:
                    end.append(j + 1)
            reg_start_list.append(start)
            reg_end_list.append(end)

        for i in range(num_agent):
            count = 0
            for j in range(gt.shape[1] - 1):
                if reg_mask_n[i, j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt_n[i, start_id]
                    end_pt = gt_n[i, end_id]
                    gt_n[i, j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j - start_id)
                    if j == end_id - 1:
                        count += 1

        if self.dataset == 'argoverse_v2' or self.dataset == 'waymo':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        valid_mask_eval = reg_mask[eval_mask]

        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]
        gt_his_eval = gt_his[eval_mask]

        marginal_trajs = traj_refine[eval_mask, :, :, :2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0), self.num_modes, -1)

        marginal_mode = torch.matmul((marginal_trajs - self.s_mean.unsqueeze(1)).permute(1, 0, 2),
                                     self.VT_k.unsqueeze(0).repeat(self.num_modes, 1, 1))
        marginal_mode = marginal_mode.permute(1, 0, 2)

        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)
        marg_mean = marginal_mode.mean(dim=1)

        marg_std = marginal_mode.std(dim=1) + self.std_reg

        mean = marg_mean
        std = marg_std

        self.joint_diffusion.eval()

        num_samples = self.num_eval_samples

        start_data = None
        reverse_steps = None
        pred_modes = self.joint_diffusion.sample(num_samples, data=data, scene_enc=scene_enc,
                                                 mean=mean, std=std, mm=marginal_mode,
                                                 mmscore=pi.exp()[eval_mask], sampling=self.sampling,
                                                 stride=self.sampling_stride, eval_mask=eval_mask,
                                                 start_data=start_data, reverse_steps=reverse_steps, num_sampling_steps=self.num_sampling_steps)

        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()


        unnorm_pred_modes = self.unnormalize(pred_modes, self.latent_mean, self.latent_std)

        rec_traj = torch.matmul(unnorm_pred_modes.permute(1, 0, 2),
                                (self.V_k).unsqueeze(0).repeat(self.num_eval_samples, 1, 1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1, 0, 2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1), self.num_future_steps, 2)

        mode_diff = rec_traj[:, :, -1, :2].unsqueeze(-2).repeat(1, 1, self.num_modes, 1) - traj_refine[eval_mask, :, -1, :2].unsqueeze(1)
        mode_diff = mode_diff.norm(dim=-1)
        mode_joint_best = torch.argmin(mode_diff, dim=-1)

        device = mean.device
        batch_idx = data['agent']['batch'][eval_mask]

        #num_scenes and num_agents_per_scene are used only for visualization
        num_scenes = batch_idx[-1].item() + 1
        num_agents_per_scene = mode_joint_best.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                      rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)


        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJADE_diff_c.update(batch_agent_idx=batch_agent_idx,
                                   pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)
        self.log('val_minJADE_diff_c', self.minJADE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minJFDE_diff_c.update(batch_agent_idx=batch_agent_idx,
                                   pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)

        self.log('val_minJFDE_diff_c', self.minJFDE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minADE_diff_c.update(      pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)
        self.log('val_minADE_diff_c', self.minADE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minFDE_diff_c.update(      pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)

        self.log('val_minFDE_diff_c', self.minFDE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.MR_c.update(      pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)

        self.log('val_MR_c', self.MR_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)


        self.CR_c.update(batch_agent_idx=batch_agent_idx, pred=rec_traj_world, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                         valid_mask=valid_mask_eval)

        self.log('val_CR_c', self.CR_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Brier.update(pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                  valid_mask=valid_mask_eval)

        self.log('val_brier_diff_c', self.Brier, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)



        if print_flag:
            print('ADE_c', self.minADE_diff_c.compute())
            print('FDE_c', self.minFDE_diff_c.compute())
            print('Brier_c', self.Brier.compute())
            print('MR_c', self.MR_c.compute())
            print('CR_c', self.CR_c.compute())

        plot = False #Switch to True for visualization

        #Plot settings
        if plot:
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                          rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

            marginal_trajs = traj_refine[eval_mask, :, :, :2]
            marg_traj_world = torch.matmul(marginal_trajs[:, :, :, :2],
                                           rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

            marg_traj_world = marg_traj_world.detach().cpu().numpy()


            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                         rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
            gt_eval_world = gt_eval_world.detach().cpu().numpy()


            gt_his_eval_world = gt_his_eval[:, :, :2]
            gt_his_eval_world = gt_his_eval_world.detach().cpu().numpy()

            img_folder = 'visual'
            sub_folder = 'folder_name' #Change to reflect your folder name
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i + 1])

                if end_id - start_id == 1:
                    continue

                temp = gt_eval[start_id:end_id]
                temp_start = temp[:, 0, :]
                temp_end = temp[:, -1, :]
                norm = torch.norm(temp_end - temp_start, dim=-1)
                if torch.max(norm) < 10:
                    continue

                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path(
                    '/path/to/av2/data/val/raw')  # Change to reflect folder (train/val/test)
                scenario_folder = base_path_to_data / scenario_id

                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)

                viz_output_dir = Path(img_folder) / sub_folder
                os.makedirs(viz_output_dir, exist_ok=True)

                viz_save_path = viz_output_dir / ('b' + str(data_batch) + '_s' + str(i) + '_' + self.sampling + '.jpg')

                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['gt_his'] = gt_his_eval_world[start_id:end_id]

                additional_traj['marg_traj'] = marg_traj_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]

                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gt_his'] = True
                traj_visible['gt_goal'] = False
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True

                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)

    def test_step(self,
                  data,
                  batch_idx):

        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True

        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        traj_his = data['agent']['position'][:, :self.num_historical_steps, :]
        gt_his = torch.cat([traj_his[..., :self.output_dim], traj_his[..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_refine[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)

        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0, 1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)

        eval_mask = data['agent']['category'] >= 2

        mask = (data['agent']['category'] >= 2) & (reg_mask[:, -1] == True) & (reg_mask[:, 0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        gt_n[0, :, :] = (gt_n[0, :, :] - gt_n[0, 0:1, :]) / 4 * 3 + gt_n[0, 0:1, :]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(gt.shape[1] - 1):
                if reg_mask_n[i, j] == True and reg_mask_n[i, j + 1] == False:
                    start.append(j)
                elif reg_mask_n[i, j] == False and reg_mask_n[i, j + 1] == True:
                    end.append(j + 1)
            reg_start_list.append(start)
            reg_end_list.append(end)

        for i in range(num_agent):
            count = 0
            for j in range(gt.shape[1] - 1):
                if reg_mask_n[i, j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt_n[i, start_id]
                    end_pt = gt_n[i, end_id]
                    gt_n[i, j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j - start_id)
                    if j == end_id - 1:
                        count += 1

        if self.dataset == 'argoverse_v2' or self.dataset == 'waymo':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        valid_mask_eval = reg_mask[eval_mask]

        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]
        gt_his_eval = gt_his[eval_mask]

        marginal_trajs = traj_refine[eval_mask, :, :, :2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0), self.num_modes, -1)

        marginal_mode = torch.matmul((marginal_trajs - self.s_mean.unsqueeze(1)).permute(1, 0, 2),
                                     self.VT_k.unsqueeze(0).repeat(self.num_modes, 1, 1))
        marginal_mode = marginal_mode.permute(1, 0, 2)

        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)
        marg_mean = marginal_mode.mean(dim=1)

        marg_std = marginal_mode.std(dim=1) + self.std_reg

        mean = marg_mean
        std = marg_std

        self.joint_diffusion.eval()

        num_samples = self.num_eval_samples

        start_data = None
        reverse_steps = None
        pred_modes = self.joint_diffusion.sample(num_samples, data=data, scene_enc=scene_enc,
                                                 mean=mean, std=std, mm=marginal_mode,
                                                 mmscore=pi.exp()[eval_mask], sampling=self.sampling,
                                                 stride=self.sampling_stride, eval_mask=eval_mask,
                                                 start_data=start_data, reverse_steps=reverse_steps,
                                                 num_sampling_steps=self.num_sampling_steps)

        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()
        unnorm_pred_modes = self.unnormalize(pred_modes, self.latent_mean, self.latent_std)

        rec_traj = torch.matmul(unnorm_pred_modes.permute(1, 0, 2),
                                (self.V_k).unsqueeze(0).repeat(self.num_eval_samples, 1, 1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1, 0, 2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1), self.num_future_steps, 2)

        mode_diff = rec_traj[:, :, -1, :2].unsqueeze(-2).repeat(1, 1, self.num_modes, 1) - traj_refine[eval_mask, :, -1, :2].unsqueeze(1)
        mode_diff = mode_diff.norm(dim=-1)
        mode_joint_best = torch.argmin(mode_diff, dim=-1)

        device = mean.device
        batch_idx = data['agent']['batch'][eval_mask]

        # num_scenes and num_agents_per_scene are used only for the visualization
        num_scenes = batch_idx[-1].item() + 1
        num_agents_per_scene = mode_joint_best.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                      rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJADE_diff_c.update(batch_agent_idx=batch_agent_idx,
                                   pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)
        self.log('test_minJADE_diff_c', self.minJADE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minJFDE_diff_c.update(batch_agent_idx=batch_agent_idx,
                                   pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                   valid_mask=valid_mask_eval)

        self.log('test_minJFDE_diff_c', self.minJFDE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minADE_diff_c.update(pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                  valid_mask=valid_mask_eval)
        self.log('test_minADE_diff_c', self.minADE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.minFDE_diff_c.update(pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                  valid_mask=valid_mask_eval)

        self.log('test_minFDE_diff_c', self.minFDE_diff_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.MR_c.update(pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                         valid_mask=valid_mask_eval)

        self.log('test_MR_c', self.MR_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.CR_c.update(batch_agent_idx=batch_agent_idx, pred=rec_traj_world, target=gt_eval[..., :self.output_dim],
                         prob=pi_eval,
                         valid_mask=valid_mask_eval)

        self.log('test_CR_c', self.CR_c, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Brier.update(pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)

        self.log('test_brier_diff_c', self.Brier, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        if print_flag:
            print('ADE_c', self.minADE_diff_c.compute())
            print('FDE_c', self.minFDE_diff_c.compute())
            print('Brier_c', self.Brier.compute())
            print('MR_c', self.MR_c.compute())
            print('CR_c', self.CR_c.compute())

        plot = False  # Switch to True for visualization

        # Plot settings
        if plot:
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                          rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

            marginal_trajs = traj_refine[eval_mask, :, :, :2]
            marg_traj_world = torch.matmul(marginal_trajs[:, :, :, :2],
                                           rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

            marg_traj_world = marg_traj_world.detach().cpu().numpy()

            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                         rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
            gt_eval_world = gt_eval_world.detach().cpu().numpy()

            gt_his_eval_world = gt_his_eval[:, :, :2]
            gt_his_eval_world = gt_his_eval_world.detach().cpu().numpy()

            img_folder = 'visual'
            sub_folder = 'folder_name' #Change to reflect your folder name
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i + 1])

                if end_id - start_id == 1:
                    continue

                temp = gt_eval[start_id:end_id]
                temp_start = temp[:, 0, :]
                temp_end = temp[:, -1, :]
                norm = torch.norm(temp_end - temp_start, dim=-1)
                if torch.max(norm) < 10:
                    continue

                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path(
                    '/path/to/av2/data/test/raw')  # Change to reflect folder (train/val/test)
                scenario_folder = base_path_to_data / scenario_id

                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)

                viz_output_dir = Path(img_folder) / sub_folder
                os.makedirs(viz_output_dir, exist_ok=True)

                viz_save_path = viz_output_dir / ('b' + str(data_batch) + '_s' + str(i) + '_' + self.sampling + '.jpg')

                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['gt_his'] = gt_his_eval_world[start_id:end_id]

                additional_traj['marg_traj'] = marg_traj_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]

                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gt_his'] = True
                traj_visible['gt_goal'] = False
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True

                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)

    def on_test_end(self):
        if self.dataset == 'argoverse_v2' or self.dataset == 'waymo':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    def set_opt_lr(self, lr):
        [optimizer], [scheduler] = self.optimizers()
        for g in optimizer.param_groups:
            g['lr'] = 0.001
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument('--qcnet_ckpt_path', type=str, required=True)
        parser.add_argument('--num_denoiser_layers', type=int, default=3)
        parser.add_argument('--num_diffusion_steps', type=int, default=10)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--diff_type', choices=['opd', 'cm', 'ict', 'ict-mm', 'ect-mm', 'ict-mm-fusion', 'ect-mm-fusion'])
        parser.add_argument('--sampling', choices=['ddpm', 'ddim'])
        parser.add_argument('--sampling_stride', type=int, default=20)
        parser.add_argument('--num_eval_samples', type=int, default=6)
        parser.add_argument('--eval_mode_error_2', type=int, default=1)
        parser.add_argument('--choose_best_mode', choices=['FDE', 'ADE'], default='ADE')
        parser.add_argument('--train_agent', choices=['all', 'eval'], default='all')
        parser.add_argument('--path_pca_s_mean', type=str, default='pca/s_mean_10.npy')
        parser.add_argument('--path_pca_VT_k', type=str, default='pca/VT_k_10.npy')
        parser.add_argument('--path_pca_latent_mean', type=str, default='pca/latent_mean_10.npy')
        parser.add_argument('--path_pca_latent_std', type=str, default='pca/latent_std_10.npy')
        parser.add_argument('--m_dim', type=int, default=10)

        return parent_parser