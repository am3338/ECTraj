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
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4'

from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset, WaymoDataset
from predictors import QCNet, DiffNet
from transforms import TargetBuilder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['argoverse_v2', 'waymo'], default='argoverse_v2')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default="4,")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--sampling', choices=['ddpm','ddim'],default='ddpm')
    parser.add_argument('--sampling_stride', type = int, default = 20)
    parser.add_argument('--num_eval_samples', type = int, default = 6)
    parser.add_argument('--eval_mode_error_2', type = int, default = 1)

    parser.add_argument('--std_reg',type = float, default=0.1)
    parser.add_argument('--path_pca_V_k', type = str,default = 'none')
    
    parser.add_argument('--network_mode', choices=['val', 'test'],default = 'val')
    parser.add_argument('--submission_file_name', type=str, default='submission')

    parser.add_argument('--num_sampling_steps', type = int, default = 1)

    args = parser.parse_args()

    model = {
        'DiffNet': DiffNet,
    }['DiffNet'].load_from_checkpoint(checkpoint_path=args.ckpt_path)


    model.add_extra_param(args)
    
    model.sampling = args.sampling
    model.sampling_stride = args.sampling_stride
    model.check_param()
    model.num_eval_samples = args.num_eval_samples
    model.eval_mode_error_2 = args.eval_mode_error_2

    model_size = 0

    for param in model.parameters():
        model_size += param.numel()

    if args.dataset == 'argoverse_v2':
        val_dataset = {
            'argoverse_v2': ArgoverseV2Dataset,
        }[model.dataset](root=args.root, split=args.network_mode,
                         transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))

    elif args.dataset == 'waymo':
        val_dataset = {
            'waymo': WaymoDataset,
        }[model.dataset](root=args.root, split=args.network_mode,
                         transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))

    else:
        raise ValueError(f'{args.dataset} is not a valid dataset')

    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    if args.network_mode == 'val':
        trainer.validate(model, dataloader)
        
        
    elif args.network_mode == 'test':
        trainer.test(model, dataloader)