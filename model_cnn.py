import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class EZVSL(nn.Module):
    def __init__(self, tau, dim):
        super(EZVSL, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(weights="ResNet18_Weights.DEFAULT")
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, 2048, kernel_size=(1, 1))

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, 2048)

        #self.aud_emb_proj = nn.Linear(2048, dim)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_proj(img)
        img = nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio) #in(N,1,257,276) out(N,512)
        aud = self.aud_proj(aud) #in#(N,512) out(N,2048)
        aud = nn.functional.normalize(aud, dim=1)

        #aud_emb_proj = self.aud_emb_proj(aud_embedding)
        #aud_emb_proj = nn.functional.normalize(aud_emb_proj, dim=1)

        return img, aud
