#train 的数据
import csv
import os
image_sub_path3 = []
image_sub_label3 = []
with open('/root/autodl-tmp/driver/train1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if int(row[2]) == 2:
            # image_sub_path3.append(os.path.join("/root/train_3",row[0].split("/")[5]+"/"+row[0].split("/")[6]))
            image_sub_path3.append(row[0])
            image_sub_label3.append(row[1])

image_sub_valid_path3 = []
image_sub_valid_label3 = []
with open('/root/autodl-tmp/driver/valid1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if int(row[2]) == 2:
            # image_sub_valid_path3.append(os.path.join("/root/valid_3",row[0].split("/")[5]))
            image_sub_valid_path3.append(row[0])
            image_sub_valid_label3.append(row[1])
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision
# 定义数据预处理方式
transform = T.Compose([
    T.Resize((96,96), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class MyDataset(Dataset):
    def __init__(self, image_path, image_label, transform=None):
        self.image_path = image_path
        self.image_label = image_label
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image_label = self.image_label[idx]
        # print("1111111111",type(image_label))
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            image_label = torch.tensor(int(image_label))

        return image, image_label
#训练
train_data_3 = MyDataset(image_sub_path3,image_sub_label3,transform=transform)
train_loader_3 = DataLoader(train_data_3, batch_size=32, shuffle=True, num_workers=8)
valid_data_3 = MyDataset(image_sub_valid_path3,image_sub_valid_label3,transform=transform)
valid_loader_3 = DataLoader(valid_data_3, batch_size=32, shuffle=True, num_workers=8)
#搭建模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
import os
from  attention import cbam_block

# 定义子网络3
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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
        #4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 8)
        #3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 8)
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



# 设置超参数
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 实例化模型并且移动到GPU
model = Network().to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # 损失函数：损失函数交叉熵

# 定义训练过程
def train(model, device, train_loader,optimizer, EPOCHS):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        # print(loss)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),100. * (batch_idx + 1) / len(train_loader), loss.item()))
    # torch.save(model.state_dict(), os.path.join("/root/autodl-tmp/driver/weight3", 'epoch_{}.pth'.format(epoch)))
# 定义测试过程
def test(model, device, valid_loader):
    correct = 0
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()  # 统计正确预测的数量
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算总的损失值
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

            # 输出valid结果
        valid_loss /= len(valid_loader.dataset)  # 计算平均损失值

        print('\nTest set: Average loss: {:.4f}, Accuracy:  {:.0f}%\n'.format(
                valid_loss,100. * correct / len(valid_loader.dataset)))
    return 100. * correct / len(valid_loader.dataset)

#定义训练最优参数
best_weight = copy.deepcopy(model.state_dict())
best_epoch = 0
best_correct = 0
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader_3, optimizer, epoch)
    avg_correct = test(model, DEVICE, valid_loader_3)
    if best_correct < avg_correct:
        best_epoch = epoch
        best_correct = avg_correct
        best_weight = copy.deepcopy(model.state_dict())
print('best_epoch {}, best_correct {:.3f}'.format(best_epoch, best_correct))
torch.save(best_weight,  os.path.join("/root/autodl-tmp/driver/weight3", 'best4.pth'))