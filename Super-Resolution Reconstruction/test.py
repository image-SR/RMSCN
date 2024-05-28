import PIL.Image
from PIL import Image
from torchvision import transforms as T
from CNN3.model import RMSCN
import torch
import os
import numpy as np
from tool import denormalize, convert_rgb_to_y, calculate_psnr
from config import opt
import cv2
import h5py


def resize(img, h, w):
    trans = T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)
    return trans(img)


def real_test(path):
    img = Image.open(path).convert('RGB')
    # w, h = img.size
    # y, b, r = img.split()
    # b = resize(b, h*4, w*4)
    # r = resize(r, h*4, w*4)
    input_image = T.ToTensor()(img).unsqueeze(0)
    model = RMSCN()
    model_state_dic = torch.load('/root/autodl-tmp/CNN3/x2/best.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input_image = input_image.cuda()
        model = model.cuda()
    with torch.no_grad():
        out = model(input_image).clamp(0.0, 1.0)
    out = out.squeeze(0)
    out = T.ToPILImage()(out)

    return out 


def test(path, scale):
    
    gt = Image.open(path).convert('RGB')
    w, h = gt.size
    print(w,h)
    w, h = (w // scale) * scale, (h // scale) * scale
    img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
    lr = img.resize((w // scale, h // scale), resample=PIL.Image.BICUBIC)
    input_image = T.ToTensor()(lr).unsqueeze(0)
    model = RMSCN() ###
    model_state_dic = torch.load('/root/autodl-tmp/CNN3/x4/best.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input_image = input_image.cuda()
        model = model.cuda()
    with torch.no_grad():
        out = model(input_image).clamp(0.0, 1.0)
    out = out.squeeze(0)
    out = T.ToPILImage()(out)
    # lr.show()
    # out.show()
    out.save("/root/autodl-tmp/photo/00261_10.png")  ##



def PSNRRGB(root,scale):

    pth_path = '/root/autodl-tmp/CNN3/x4/best.pth'###

    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = RMSCN()###
    model_state_dic = torch.load(pth_path)
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        lr = to_tensor(lr).unsqueeze(0)
        #print("1111",lr.shape)
        if opt.cuda:
            lr = lr.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(lr).squeeze(0)
        labels = to_tensor(img)
        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))
        
        preds = preds[scale:-scale, scale:-scale]
        labels = labels[scale:-scale, scale:-scale]

        res += calculate_psnr(preds, labels)
    print('PSNR:{:.3f}'.format(res/len(img_paths)))
    return (res/len(img_paths))
    
def SSIM(root, scale):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = RMSCN()###
    model_state_dic = torch.load('/root/autodl-tmp/CNN3/x4/best.pth')###
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        input_image = to_tensor(lr).unsqueeze(0)
        if opt.cuda:
            input_image = input_image.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(input_image).squeeze(0)
        labels = to_tensor(img)

        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))

        preds = preds.numpy()
        labels = labels.numpy()

        res += calculate_ssim(preds, labels)

    print('SSIM:{:.4f}'.format(res / len(img_paths)))



def calculate_ssim(img1, img2):

    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

