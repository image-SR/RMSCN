import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1(nn.Module):
    def __init__(self, inChannals, outChannals):
        
        super(ResBlock1, self).__init__()

        
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=7, stride=1, padding=3)
        
        self.conv1_1 = nn.Conv2d(64 *4, 64 *2 , kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(64 *2, 64, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x_1 = self.conv1(x)
        x_1 = self.relu(x_1)
        
        x_2 = self.conv2(x)
        x_2 = self.relu(x_2)
        
        x_3 = self.conv3(x)
        x_3 = self.relu(x_3)
        
        x_ = torch.cat((x_1,x_2,x_3,x), 1)

        x_ = self.conv1_1(x_)
        x_ = self.conv1_2(x_)

        
        x_4 = self.conv1(x_)
        x_4 = self.relu(x_4)
        
        x_5 = self.conv2(x_)
        x_5 = self.relu(x_5)
        
        x_6 = self.conv3(x_)
        x_6 = self.relu(x_6)
        
        x_ = torch.cat((x_4,x_5,x_6,x), 1)

        x_ = self.conv1_1(x_)
        x_ = self.conv1_2(x_)

        x = x + x_
        
        return x
