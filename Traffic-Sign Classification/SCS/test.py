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
from  attention import cbam_block

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,1,1)#96   24
        self.max_pool1 = nn.MaxPool2d(2)#48
        self.conv2 = nn.Conv2d(32, 32, 3,1,1)#48  12
        self.conv3 = nn.Conv2d(32, 8, 3,1,1)#48  12
        #2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 3)
        #3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 3)
        #4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 3)
        
        # self.fc1 = nn.Linear(1152, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        # print("11111",x.shape)
        x = self.max_pool1(x)
        
        x1 = x
        
        # print("2222222",x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        x =x1 + x
            
        x = self.conv3(x)
        x = F.relu(x)

        # print("11111",x.shape)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
class Network_1(nn.Module):
    def __init__(self):
        super(Network_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3,1,1) #96   24
        self.dropout1 = nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool2d(2)  #48    12
        self.conv2 = nn.Conv2d(64, 64, 3,1,1) #48  12
        # self.dropout2 = nn.Dropout(0.5)
        # self.max_pool2 = nn.MaxPool2d(2)  #24
        self.conv3 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv4 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv5 = nn.Conv2d(64, 64, 3,1,1)#48  12 
        
        self.conv6 = nn.Conv2d(64, 8, 3,1,1)#48   12
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.conv7 = nn.Conv2d(32, 16, 3,1,1)
        #4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 19)
        #2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 19)
        #3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 19)
        #
        # self.fc1 = nn.Linear(1152, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 19)
    def forward(self, x):
        in_size = x.size(0)        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.max_pool1(x)
        
        x1 = x
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        
        # x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        
        x = x1+x
        
        x = self.conv6(x)
        x = F.relu(x)
        # x = self.max_pool1(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
class Network_2(nn.Module):
    def __init__(self):
        super(Network_2, self).__init__()
        #注意力
        self.cbam_block = cbam_block(64)
        
        self.conv1 = nn.Conv2d(3, 64, 3,1,1) #96   24
        self.dropout1 = nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool2d(2)  #48   12
        self.conv2 = nn.Conv2d(64, 64, 3,1,1) #48   12
        # self.dropout2 = nn.Dropout(0.5)
        # self.max_pool2 = nn.MaxPool2d(2)  #24
        self.conv3 = nn.Conv2d(64, 64, 3,1,1)#48  12 
        # self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv5 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv6 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv7 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv8 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv9 = nn.Conv2d(64, 64, 3,1,1)#48  12
        
        self.conv10 = nn.Conv2d(64, 8, 3,1,1)#48  12
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.conv7 = nn.Conv2d(32, 16, 3,1,1)
        #2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 16)
        #3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 16)
        #4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 16)
        # self.fc1 = nn.Linear(1152, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 16)
    def forward(self, x):
        in_size = x.size(0)        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.max_pool1(x)
        
        x1 = x
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        
        # x = self.max_pool2(x)
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
        
        x = x1+x
        
        x = self.conv10(x)
        x = F.relu(x)
        # x = self.max_pool1(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
class Network_3(nn.Module):
    def __init__(self):
        super(Network_3, self).__init__()
        #注意力
        self.cbam_block = cbam_block(64)
        
        self.conv1 = nn.Conv2d(3, 64, 3,1,1) #96   24
        self.dropout1 = nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool2d(2)  #48   12
        self.conv2 = nn.Conv2d(64, 64, 3,1,1) #48  12
        # self.dropout2 = nn.Dropout(0.5)
        # self.max_pool2 = nn.MaxPool2d(2)  #24  
        self.conv3 = nn.Conv2d(64, 64, 3,1,1)#48   12
        # self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv5 = nn.Conv2d(64, 64, 3,1,1)#48  12
        self.conv6 = nn.Conv2d(64, 8, 3,1,1)#48  12
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.max_pool1 = nn.MaxPool2d(2)
        # self.conv7 = nn.Conv2d(32, 16, 3,1,1)
        #2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 8)
        #3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 8)
        #4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 8)
        #
        # self.fc1 = nn.Linear(1152, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 8)
    def forward(self, x):
        in_size = x.size(0)        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.max_pool1(x)
        
        x1 = x
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.max_pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        # x = self.dropout3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        
        x = x1+x
        
        x = self.conv6(x)
        x = F.relu(x)
        # x = self.max_pool1(x)
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
    transforms.Resize((96,96), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载训练好的模型  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(DEVICE)
model_1 = Network_1().to(DEVICE)
model_2 = Network_2().to(DEVICE)
model_3 = Network_3().to(DEVICE)
# 实例化模型并且移动到GPU
model_state_dic = torch.load('/root/autodl-tmp/driver/weight/best4.pth')
model.load_state_dict(model_state_dic)
model_state_dic_1 = torch.load('/root/autodl-tmp/driver/weight1/best4.pth')
model_1.load_state_dict(model_state_dic_1)
model_state_dic_2 = torch.load('/root/autodl-tmp/driver/weight2/best4.pth')
model_2.load_state_dict(model_state_dic_2)
model_state_dic_3 = torch.load('/root/autodl-tmp/driver/weight3/best4.pth')
model_3.load_state_dict(model_state_dic_3)
def image_test(path):
    image = Image.open(path)    
    image = transform(image).unsqueeze(0).to(DEVICE)# 添加batch维度，与训练时的输入格式一致  
    # 进行前向传播 
    model.eval()
    with torch.no_grad():  
        output = model(image)  
        _, predicted_1 = torch.max(output, 1)  # 获取最大概率对应的类别作为预测结果
        # print('aaaPredicted class:', predicted.item())
        if predicted_1.item() == 0:
            output = model_1(image)  
            _, predicted_2 = torch.max(output, 1)  # 获取最大概率对应的类别作为预测结果
            # print('Predicted class:', predicted.item())
            return predicted_1,predicted_2
        elif predicted_1.item() == 1:
            output = model_2(image)
            _, predicted_2 = torch.max(output, 1)  # 获取最大概率对应的类别作为预测结果
            return predicted_1,predicted_2
            # print('Predicted class:', predicted.item())
        elif predicted_1.item() == 2:
            output = model_3(image)
            _, predicted_2 = torch.max(output, 1)  # 获取最大概率对应的类别作为预测结果
            return predicted_1,predicted_2
            # print('Predicted class:', predicted.item())

#打开csv读取图片信息
#test是原图片
with open('/root/test1.csv', 'r') as input_file:  
    reader = csv.reader(input_file)  
    data = list(reader)
    print(len(data))
    sum_acc = 0
    for row in data:
        path = "/root/test4/"+ row[0]
        #测试
        predicted_1,predicted_2 = image_test(path)
        if int(row[1]) == predicted_1  and int(row[2]) == predicted_2:
            sum_acc = sum_acc + 1
    print("准确率:",sum_acc/len(data))    