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
from argparse import ArgumentParser

import os

import torch

import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule, WaymoDataModule
from predictors import QCNet, DiffNet
from datasets import ArgoverseV2Dataset, WaymoDataset
from transforms import TargetBuilder
from torch_geometric.loader import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int,
                        default=0)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default="1")
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)


    parser.add_argument('--std_reg', type=float, default=0.1)
    parser.add_argument('--path_pca_V_k', type=str, default='none')


    parser.add_argument('--num_sampling_steps', type = int, default = 1)
    parser.add_argument('--ema_rate', type=float, default=0.9999)


    DiffNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DiffNet(args)

    model.add_extra_param(args)

    if args.dataset == 'argoverse_v2':
        datamodule = {
            'argoverse_v2': ArgoverseV2DataModule,
        }[args.dataset](**vars(args))

    elif args.dataset == 'waymo':
        datamodule = {
            'waymo': WaymoDataModule,
        }[args.dataset](**vars(args))

    else:
        raise ValueError(f'{args.dataset} is not a valid dataset')

    model_checkpoint = ModelCheckpoint(monitor='val_minFDE_diff_c', save_top_k=5, mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         num_sanity_val_steps=1)
    trainer.fit(model, datamodule)
