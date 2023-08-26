import torch.nn as nn
import torch
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


def alpha_loss(output, target):
        crocs_entorpy = F.cross_entropy(output[0],target[0])
        mse = F.mse_loss(output[1],target[1])
        return crocs_entorpy+mse
                


class ConnNet(nn.Module):
    def __init__(self,  cols: int, rows: int):
        super().__init__()
        inside = 128

  
        self.l1 = nn.Conv2d(2,inside,3,padding=1)
        self.bn = nn.BatchNorm2d(inside)
        res_blocks = [BasicBlock(inside, inside) for i in range(5)]
        self.body = nn.Sequential(*res_blocks)

        #Policy head
        prepol = 32
        self.conv1 = nn.Conv2d(inside, prepol, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(prepol)
        self.fc = nn.Linear(cols*rows*prepol, cols)

        #Value head
        self.conv3 = nn.Conv2d(inside, 3, kernel_size=1) # value head
        self.bn3= nn.BatchNorm2d(3)
        self.fc3 = nn.Linear(3*rows*cols, 32)
        self.fc4 = nn.Linear(32, 1)

    def value_head(self, x):
        x = self.conv3(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn3(x)
        x = x.view(x.shape[0],-1)
        x = self.fc3(x)
        x = F.leaky_relu(x,0.01)
        x = self.fc4(x)
        x = torch.tanh(x)

        return x

    def policy_head(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x).softmax(dim=1)

        return x

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = F.leaky_relu(x,0.01)
        x = self.body(x)
        
        value = self.value_head(x)
        policy = self.policy_head(x)
        return policy, value



