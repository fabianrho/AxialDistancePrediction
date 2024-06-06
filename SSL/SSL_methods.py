import torch
import torch.nn as nn

import copy

from lightly.loss import NTXentLoss, NegativeCosineSimilarity, BarlowTwinsLoss,DINOLoss

from lightly.models.modules import (
    SimCLRProjectionHead,
    BYOLPredictionHead,
    BYOLProjectionHead,
    BarlowTwinsProjectionHead,
    DINOProjectionHead,
    MoCoProjectionHead
)
# transforms
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

from lightly.models.utils import deactivate_requires_grad, update_momentum

from lightly.utils.scheduler import cosine_schedule



############# https://docs.lightly.ai/self-supervised-learning/examples/models.html

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(768, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(768, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(768, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z


class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(768, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key



class AxialDistanceProjection(torch.nn.Module):
    def __init__(self, backbone, hidden_dim=768):
        super(AxialDistanceProjection, self).__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim*2, self.hidden_dim),

            torch.nn.LeakyReLU(),

            torch.nn.Linear(self.hidden_dim, 1),

            
        )

    def forward(self, x1, x2):

        x1 = self.backbone.encode(x1)
        x2 = self.backbone.encode(x2)


        x = torch.cat([x1, x2], dim=1)

        x = self.projection_head(x)
        return x


if __name__ == "__main__":
    pass