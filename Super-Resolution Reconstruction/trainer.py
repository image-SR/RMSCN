from DataSet import DataSet, ValidDataset
from config import opt
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from tool import calculate_psnr, convert_rgb_to_y, denormalize ,AverageMeter
import copy
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from CNN3.model import RMSCN


# def transfer_model(pretrain_file, model):
#     pretrain_dict = torch.load(pretrain_file)
#     model_dict = model.state_dict()
#     pretrain_dict = transfer_state_dict(pretrain_dict, model_dict)
#     model_dict.update(pretrain_dict)
#     model.load_state_dict(model_dict)
#     return model


# def transfer_state_dict(pretrain_dict, model_dict):
#     state_dict = {}
#     count = 0
#     for k, v in pretrain_dict.items():
#         if k in model_dict.keys():
#             state_dict[k] = v
#             count += 1
#     if count == 0:
#         print('no parameters update!!!')
#     else:
#         print('update successfully!!!')
#     return state_dict
opt.outputs_dir = os.path.join(opt.outputs_dir, 'x{}'.format(2))  #!!!
if not os.path.exists(opt.outputs_dir):
    os.makedirs(opt.outputs_dir)

def train():
    
    train_data = DataSet(opt.train_root)
    valid_data = ValidDataset(opt.validation_root)

    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_data_loader = DataLoader(valid_data, batch_size=1)

    
    net = RMSCN()  #!!
    
    # 计算模型的参数个数
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print("Total parameters in the model: ", total_params/1000)
 
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    
    if opt.cuda:
        net = net.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        
    best_weight = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_psnr = 0
    
    best_gap = 1
    best_epoch_gap = 0
    #vis = visdom.Visdom(env=u'aan')
    for epoch in range(opt.epoch):
        res_loss = 0
        net.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_data) - len(train_data) % opt.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1, opt.epoch))
             
            for data in train_data_loader:
                lr, hr = Variable(data[0]), Variable(data[1])
                if opt.cuda:
                    lr = lr.cuda()
                    hr = hr.cuda()
                
                sr = net(lr)
                
                loss = (1 - ((epoch+1)/opt.epoch))* criterion1(sr, hr) + ((epoch+1)/opt.epoch) * criterion2(sr, hr)
                # loss = criterion1(sr, hr)
                
                # print("loss",loss.item())
                res_loss +=loss.item()
                # print("res_loss",res_loss)
                epoch_losses.update(loss.item(), len(hr))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(hr))
                # train_loss += loss.item()
                # print('epoch:{},loss:{:.3f}'.format(epoch+1, train_loss/(ii+1)))
                
            avg_loss = res_loss/len(train_data_loader)
        train_loss.append(avg_loss)
        torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(epoch+1))) 
        net.eval()
        res = 0
        for item in valid_data_loader:
            lr, hr = item
            if opt.cuda:
                lr = lr.cuda()
                hr = hr.cuda()
            with torch.no_grad():
                sr = net(lr)
            sr = convert_rgb_to_y(denormalize(sr.squeeze(0)))
            hr = convert_rgb_to_y(denormalize(hr.squeeze(0)))
            res += calculate_psnr(sr, hr)

        
        avg_psnr = res/len(valid_data_loader)
        print('Epoch:{} eval_psnr:{:.3f}'.format(epoch+1,avg_psnr))
        
        ave = round(avg_psnr.item(),3)
        train_psnr.append(ave)

        
        if best_psnr < avg_psnr:
            best_epoch = epoch+1
            best_psnr = avg_psnr
            best_weight = copy.deepcopy(net.state_dict())
    print('best_epoch {}, best_psnr {:.3f}'.format(best_epoch, best_psnr))

    torch.save(best_weight,  os.path.join(opt.outputs_dir, 'best.pth'))


if __name__ == '__main__':
    train_loss = []
    train_psnr = []
    train()
    print("train_loss",train_loss)
    def save_loss(loss, loss_name):
        fileObject = open('/root/autodl-tmp/loss.txt', 'a')
        fileObject.write("{}".format(loss_name) + '\n')
        for ip in loss:
            fileObject.write(str(ip) + " " + ",")
        fileObject.write('\n')
        fileObject.close()
    save_loss(train_loss, "train loss")
    print("train_psnr",train_psnr)
    def save_psnr(psnr, psnr_name):
        fileObject = open('/root/autodl-tmp/psnr.txt', 'a')
        fileObject.write("{}".format(psnr_name) + '\n')
        for ip in psnr:
            fileObject.write(str(ip) + " " + ",")
        fileObject.write('\n')
        fileObject.close()
    save_psnr(train_psnr, "train psnr")