import torch
import torch.nn as nn

class Distrloss_layer(nn.Module):

    def __init__(self):
        super(Distrloss_layer,self).__init__()
        #self._channels = channels
        self.v = 0.5
    def forward(self, input):
        if input.dim() != 5 and input.dim() != 3:
            raise ValueError('expected 5D or 3D input (got {}D input)'
                             .format(input.dim()))
        #if input.size()[1] != self._channels:
        #    raise ValueError('expected {} channels (got {}D input)'
         #                    .format(self._channels, input.size()[1]))

        if input.dim() == 5:
            mean = input.mean(dim=-1).mean(dim=-1).mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)) ** 2
                    ).mean(dim=-1).mean(dim=-1).mean(dim=-1).mean(dim=0)
        elif input.dim() == 3:
            mean = input.mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2)) ** 2).mean(dim=-1).mean(dim=0)

        var = var + 1e-10 # to avoid 0 variance
        std = var.abs().sqrt()

        distrloss = ((mean - self.v) ** 2).mean()
        
        #return [distrloss1, distrloss2] 
        return distrloss
