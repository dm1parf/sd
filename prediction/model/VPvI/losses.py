import torch
from torch import nn
from torch.nn import functional as F

from .flownet import resample


class Consistency(nn.Module):
    # Consistency loss for a pair of optical flow
    def __init__(self):
        super(Consistency, self).__init__()
        self.beta = 0.05
        self.weight = 0.001

    def L2_norm(self, x):
        return F.normalize(x, p=2, dim=1, eps=1e-12)#.unsqueeze(1)

    def forward(self, flow_fwd, flow_bwd, stage_num):
        devide = flow_fwd.get_device()
        alpha = torch.FloatTensor([1.5]).cuda(devide)

        bwd2fwd_flow_pyramid = resample(flow_bwd, flow_fwd)# From bwd coordinate to src coordinate
        fwd2bwd_flow_pyramid = resample(flow_fwd, flow_bwd)# From fwd coordinate to tgt coordinate
        #print("bwd2fwd_flow_pyramid", bwd2fwd_flow_pyramid.size())
        fwd_diff = torch.abs(bwd2fwd_flow_pyramid + flow_fwd)# In src
        bwd_diff = torch.abs(fwd2bwd_flow_pyramid + flow_bwd)# In tgt
        #print("fwd_diff size = ", fwd_diff.size())
        fwd_consist_bound = self.beta * self.L2_norm(flow_fwd)
        bwd_consist_bound = self.beta * self.L2_norm(flow_bwd)
        #print("fwd_consist_bound = ", fwd_consist_bound.size())
        fwd_consist_bound = alpha.clone().detach()#torch.max(fwd_consist_bound, alpha).clone().detach()
        #bwd_consist_bound = torch.max(bwd_consist_bound, alpha).clone().detach()
        bwd_consist_bound = alpha.clone().detach()
        fwd_mask = (fwd_diff < fwd_consist_bound).float()# In src
        bwd_mask = (bwd_diff < bwd_consist_bound).float()# In tgt

        if stage_num == 2:
            flow_consistency_loss = self.weight/2 * \
                (torch.sum(torch.mean(fwd_diff, dim=1, keepdim=True) * fwd_mask)  + \
                 torch.sum(torch.mean(bwd_diff, dim=1, keepdim=True) * bwd_mask))
            #(torch.sum(torch.mean(fwd_diff, dim=1, keepdim=True))
        else:
            flow_consistency_loss = self.weight/2 *  (\
             torch.sum(torch.mean(bwd_diff, dim=1, keepdim=True)))
        return fwd_mask, bwd_mask, flow_consistency_loss
