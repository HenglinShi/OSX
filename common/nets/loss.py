import torch
import torch.nn as nn
import pdb
class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        #pdb.set_trace()
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss
    

class SilhouetteLoss(nn.Module):
    def __init__(self):
        super(SilhouetteLoss, self).__init__()
        #self.activation = nn.Sigmoid()

    def forward(self, mask_pred, mask_gt):
        #loss = torch.abs(coord_out - coord_gt) * valid
        loss = torch.abs(mask_pred - mask_gt)
        #pdb.set_trace()
        #if is_3D is not None:
        #    loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
        #    loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, out, gt_index):
        loss = self.ce_loss(out, gt_index)
        return loss
