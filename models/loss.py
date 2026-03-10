import torch
from torch import nn


class RSD_Criterion(nn.Module):
    def __init__(self, first_prune_epoch, second_prune_epoch, windows):
        super(RSD_Criterion, self).__init__()
        self.first_prune_epoch = first_prune_epoch
        self.second_prune_epoch = second_prune_epoch
        self.windows = windows

    def forward(self, pred_point, gt_point, batch_size, epoch_nums, idx, values):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1)  # [B*N]
        loss = loss_map.detach().clone().view(batch_size, -1)   # [B, N]
        if self.first_prune_epoch <= epoch_nums < (self.first_prune_epoch + self.windows):
            current_epoch = epoch_nums - self.first_prune_epoch
            values[idx, current_epoch] = torch.median(loss, dim=1).values
        elif self.second_prune_epoch <= epoch_nums < (self.second_prune_epoch + self.windows):
            current_epoch = epoch_nums - self.second_prune_epoch
            values[idx, current_epoch] = torch.median(loss, dim=1).values
        loss_map = torch.mean(loss_map)

        # loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        # if self.first_prune_epoch <= epoch_nums < (self.first_prune_epoch + self.windows):
        #     loss_lw = loss_map.detach().clone()
        #     current_epoch = epoch_nums - self.first_prune_epoch
        #     # print(batch_nums.shape)
        #     # print(loss_map.shape)
        #     for i in range(batch_size):
        #         # 取出当前序列中的第N个batch数据
        #         mask = batch_nums == i
        #         values[idx[i], current_epoch] = torch.median(loss_lw[mask])  # 将其求中值，以忽略噪点影响
        # elif self.second_prune_epoch <= epoch_nums < (self.second_prune_epoch + self.windows):
        #     loss_lw = loss_map.detach().clone()
        #     current_epoch = epoch_nums - self.second_prune_epoch
        #     for i in range(batch_size):
        #         # 取出当前序列中的第N个batch数据
        #         mask = batch_nums == i
        #         values[idx[i], current_epoch] = torch.median(loss_lw[mask])  # 将其求中值，以忽略噪点影响

        # loss_map = torch.mean(loss_map)

        return loss_map, values


class REG_Criterion(nn.Module):
    def __init__(self):
        super(REG_Criterion, self).__init__()

    def forward(self, pred_point, gt_point):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        return loss_map

class CriterionPose(nn.Module):
    def __init__(self, rot_weight=6):
        super(CriterionPose, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()
        self.rot_weight = rot_weight
    def forward(self, pred_t, pred_q, gt_t, gt_q):
        loss_t = self.t_loss_fn(pred_t, gt_t)
        loss_q = self.q_loss_fn(pred_q, gt_q)
        loss = 1 * loss_t + self.rot_weight * loss_q
        return loss


def barlow_twins_loss(embeddings_A, embeddings_B, lambda_param=0.005):
    '''
    embeddings_A: (N, D)
    embeddings_B: (N, D)
    '''
    batch_size = embeddings_A.size(0)

    embeddings_A = (embeddings_A - embeddings_A.mean(0)) / (
        embeddings_A.std(0) + 1e-7
    )
    embeddings_B = (embeddings_B - embeddings_B.mean(0)) / (
        embeddings_B.std(0) + 1e-7
    )

    cross_correlation = torch.mm(embeddings_A.T, embeddings_B) / batch_size

    on_diag = torch.diagonal(cross_correlation).add_(-1).pow(2).sum()
    off_diag = (
        cross_correlation.pow(2).sum()
        - torch.diagonal(cross_correlation).pow(2).sum()
    )

    loss = on_diag + lambda_param * off_diag
    return loss
