#train 的数据
import csv
image_path = []
image_label = []

import os
# 读取CSV文件并提取第一列数据
with open('/root/autodl-tmp/driver/train1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print(type(row[0]))
        # print(type(row[2]))
        # image_path.append(os.path.join("/root/train_4",row[0].split("/")[5]+"/"+row[0].split("/")[6]))
        image_path.append(row[0])
        image_label.append(row[2])

#valid 的数据
import csv
image_valid_path = []
image_valid_label = []
import csv
# 打开CSV文件并读取数据
with open('/root/autodl-tmp/driver/valid1.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    for row in data:
        # print()
        # image_valid_path.append(os.path.join("/root/valid_4",row[0].split("/")[5]))
        image_valid_path.append(row[0])
        image_valid_label.append(row[2])
        
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
train_data = MyDataset(image_path,image_label,transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)

#测试
valid_data = MyDataset(image_valid_path,image_valid_label,transform=transform)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True, num_workers=8)


#搭建模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
import os
#网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,1,1)#96   24
        self.max_pool1 = nn.MaxPool2d(2)#48
        self.conv2 = nn.Conv2d(32, 32, 3,1,1)#48  12
        self.conv3 = nn.Conv2d(32, 8, 3,1,1)#48  12
        #×4
        self.fc1 = nn.Linear(18432, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 3)
        #×2
        # self.fc1 = nn.Linear(4608, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 3)
        #×3
        # self.fc1 = nn.Linear(10368, 1024)
        # self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(64, 3)
        #
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
    # torch.save(model.state_dict(), os.path.join("/root/autodl-tmp/driver/weight", 'epoch_{}.pth'.format(epoch)))
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
    train(model, DEVICE, train_loader, optimizer, epoch)
    avg_correct = test(model, DEVICE, valid_loader)
    if best_correct < avg_correct:
        best_epoch = epoch
        best_correct = avg_correct
        best_weight = copy.deepcopy(model.state_dict())
print('best_epoch {}, best_correct {:.3f}'.format(best_epoch, best_correct))
torch.save(best_weight,  os.path.join("/root/autodl-tmp/driver/weight", 'best4.pth'))
