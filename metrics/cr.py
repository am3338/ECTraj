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
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter

class CR(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(CR, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               miss_criterion: str = 'FDE',
               collision_threshold: float = 1.0) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        #pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred

        scenarios = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in scenarios:
                scenarios[sc_id] = pred[i, :, :, :].unsqueeze(0)
            else:
                scenarios[sc_id] = torch.cat([scenarios[sc_id], pred[i, :, :, :].unsqueeze(0)], dim=0)

        for sc in scenarios:
            fde = torch.norm(scenarios[sc][torch.arange(scenarios[sc].size(0)), :, -1] -
                             target[torch.arange(scenarios[sc].size(0)), -1].unsqueeze(-2),
                             p=2, dim=-1)
            min_fde_index = torch.argmin(fde.mean(dim=0))
            best_pred = scenarios[sc][:, min_fde_index, :, :]
            diff = best_pred[:, None, :, :] - best_pred[None, :, :, :]

            dist = torch.linalg.norm(diff, dim=-1)
            dist = dist.masked_fill(torch.eye(scenarios[sc].shape[0], device=pred_topk.device).bool()[:, :, None],
                                    float('inf'))

            cr = (dist < collision_threshold).any(dim=(1, 2))
            collision_rate_for_scenario = cr.float().mean()
            diff_gt = target[:, None, :, :] - target[None, :, :, :]
            dist_gt = torch.linalg.norm(diff_gt)
            dist_gt = dist_gt.masked_fill(torch.eye(scenarios[sc].shape[0], device=pred_topk.device).bool()[:, :, None],
                                    float('inf'))
            cr_gt = (dist_gt < collision_threshold).any(dim=(1, 2))
            collision_rate_for_gt = cr_gt.float().mean()

            self.sum += collision_rate_for_scenario
        self.count += len(scenarios)


    def compute(self) -> torch.Tensor:
        return self.sum / self.count
