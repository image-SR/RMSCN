import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision
import torch  
import torchvision.transforms as transforms  
from PIL import Image  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
import os
import csv
# 定义网络

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3,1,1) #96   24
        self.dropout1 = nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool2d(2)  #48   12
        self.conv2 = nn.Conv2d(64, 64, 3,1,1) #48   12
        self.conv3 = nn.Conv2d(64, 64, 3,1,1)#48  12 
        self.conv4 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv5 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv6 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv7 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv8 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv9 = nn.Conv2d(64, 64, 3,1,1)#48  12
        
        self.conv10 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv11 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv12 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv13 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv14 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv15 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv16 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv17 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv18 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv19 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv20 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv21 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv22 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv23 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv24 = nn.Conv2d(64, 64, 3,1,1)#48  12
        
        self.conv25 = nn.Conv2d(64, 8, 3,1,1)#48  12
        #×2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 43)
        #×3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 43)
        #×4
        # self.fc1 = nn.Linear(18432, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 43)
        
        self.fc1 = nn.Linear(1152, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 43)

    def forward(self, x):
        in_size = x.size(0)        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.max_pool1(x)
        
        x1 = x
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        
        x = self.conv10(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        x = self.conv14(x)
        x = F.relu(x)
        x = self.conv15(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = F.relu(x)
        x = self.conv17(x)
        x = F.relu(x)
        x = self.conv18(x)
        x = F.relu(x)
        x = self.conv19(x)
        x = F.relu(x)
        x = self.conv20(x)
        x = F.relu(x)
        x = self.conv21(x)
        x = F.relu(x)
        x = self.conv22(x)
        x = F.relu(x)
        x = self.conv23(x)
        x = F.relu(x)
        x = self.conv24(x)
        x = F.relu(x)
        
        x = x1+x
        
        x = self.conv25(x)
        x = F.relu(x)

        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x

transform = transforms.Compose([
    transforms.Resize((24,24), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载训练好的模型  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(DEVICE)
# 实例化模型并且移动到GPU
model_state_dic = torch.load('/root/autodl-tmp/driver/all_weight/best.pth')###
model.load_state_dict(model_state_dic)
def image_test(path):
    image = Image.open(path)    
    image = transform(image).unsqueeze(0).to(DEVICE)# 添加batch维度，与训练时的输入格式一致  
    # 进行前向传播 
    model.eval()
    with torch.no_grad():  
        output = model(image)  
        _, predicted = torch.max(output, 1)  # 获取最大概率对应的类别作为预测结果
        return predicted


#打开csv读取图片信息
#test原始小图片
with open('/root/all_test.csv', 'r') as input_file:  
    reader = csv.reader(input_file)  
    data = list(reader)
    print(len(data))
    sum_acc = 0
    for row in data:
        path = "/root/test/"+ row[0]
        #测试
        predicted = image_test(path)
        if int(row[1]) == predicted:
            sum_acc = sum_acc + 1
    print("准确率:",sum_acc/len(data))    