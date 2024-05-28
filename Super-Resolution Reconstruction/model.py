from torch import nn
import torch
from  CNN3.block import ResBlock1
from  CNN3.attention import cbam_block
class RMSCN(nn.Module):
    def __init__(self, num_channels=3):
        super(RMSCN, self).__init__()
        
        self.cbam_block = cbam_block(64)
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.resBlock1 = self._makeLayer_(ResBlock1, 64, 64, 1)
        
        self.conv1x3 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)

        self.convPos1 = nn.Conv2d(64, 64 * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.pixelShuffler1 = nn.PixelShuffle(2)
       
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1) 
        
    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals)) 
        return nn.Sequential(*layers)   
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        x_1_1 = self.resBlock1(x)#64    48 
        x_1_2 = self.resBlock1(x_1_1)#64  48
        x_1_3 = self.resBlock1(x_1_2)#64  48
        x_1_4 = self.resBlock1(x_1_3)#64  48
        x_1_5 = self.resBlock1(x_1_4)#64  48
        x_1_6 = self.resBlock1(x_1_5)#64  48  
        x_1_7 = self.resBlock1(x_1_6)#64  48
        x_1_8 = self.resBlock1(x_1_7)#64  48
        x_1_9 = self.resBlock1(x_1_8)#64  48
        x_1_10 = self.resBlock1(x_1_9)#64  48
        
        #
        x_a1 = torch.cat((x_1_1,x_1_2), 1)#64 * 2 
        x_a1 = self.conv1x3(x_a1)#64

        
        x_a2 = torch.cat((x_1_3,x_1_4), 1)#64 * 2 
        x_a2 = self.conv1x3(x_a2)#64

        
        x_a3 = torch.cat((x_1_5,x_1_6), 1)#64 * 2 
        x_a3 = self.conv1x3(x_a3)#64

        
        x_a4 = torch.cat((x_1_7,x_1_8), 1)#64 * 2 
        x_a4 = self.conv1x3(x_a4)#64
        x_a4 = self.cbam_block(x_a4) 
        
        x_a5 = torch.cat((x_1_9,x_1_10), 1)#64 * 2 
        x_a5 = self.conv1x3(x_a5)#64
        x_a5 = self.cbam_block(x_a5) 
        
        #
        x_a6 = torch.cat((x_a1,x_a2), 1)#64 * 2 
        x_a6 = self.conv1x3(x_a6)#64
        x_a6 = self.cbam_block(x_a6)
         
        x_a7 = torch.cat((x_a2,x_a3), 1)#64 * 2 
        x_a7 = self.conv1x3(x_a7)#64
        x_a7 = self.cbam_block(x_a7) 
        
        x_a8 = torch.cat((x_a3,x_a4), 1)#64 * 2 
        x_a8 = self.conv1x3(x_a8)#64
        x_a8 = self.cbam_block(x_a8) 
        
        x_a9 = torch.cat((x_a4,x_a5), 1)#64 * 2 
        x_a9 = self.conv1x3(x_a9)#64
        x_a9 = self.cbam_block(x_a9) 
        
        #
        x_a10 = torch.cat((x_a6,x_a7), 1)#64 * 2 
        x_a10 = self.conv1x3(x_a10)#64
        x_a10 = self.cbam_block(x_a10)
        
        x_a11 = torch.cat((x_a7,x_a8), 1)#64 * 2 
        x_a11 = self.conv1x3(x_a11)#64
        x_a11 = self.cbam_block(x_a11)
        
        x_a12 = torch.cat((x_a8,x_a9), 1)#64 * 2 
        x_a12 = self.conv1x3(x_a12)#64
        x_a12 = self.cbam_block(x_a12)
        
        #
        x_a13 = torch.cat((x_a10,x_a11), 1)#64 * 2 
        x_a13 = self.conv1x3(x_a13)#64
        x_a13 = self.cbam_block(x_a13)
        
        x_a14 = torch.cat((x_a11,x_a12), 1)#64 * 2 
        x_a14 = self.conv1x3(x_a14)#64
        x_a14 = self.cbam_block(x_a14)
        
        #
        x_a15 = torch.cat((x_a13,x_a14), 1)#64 * 2 
        x_a15 = self.conv1x3(x_a15)#64
        x_a15 = self.cbam_block(x_a15)
        

        
        x = x + x_a15 #64 48
        
        x = self.convPos1(x)#64
        x = self.pixelShuffler1(x)#64
       
        x =self.conv2(x)#3
        x = self.relu(x)
        
        return x