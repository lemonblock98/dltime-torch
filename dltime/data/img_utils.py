from pickle import load
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image
import numpy as np


def load_car_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                            std=(0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='D:/github/tsGAN-torch/car_imgs', transform=transform)
    return train_dataset



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='latin1')
    return dict_


def save_cifar_10_car_pics():
    batch = unpickle('D:/dataset/cifar-10-batches-py/data_batch_4')
    img_list = [batch['data'][i].reshape(3, 32, 32).transpose(1, 2, 0) for i in range(10000) if \
        batch['labels'][i] == 1]
    
    img_list = random.sample(img_list, 500)
    for i, img in tqdm(enumerate(img_list)):
        img = Image.fromarray(img)
        img.save('car_imgs/car%d.png' % i)


def denorm(x):
    # 这个函数的功能是把规范化之后的数据还原到原始数据
    # denormalize
    out = (x + 1) / 2
    # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内
    return out.clamp(0, 1)


def show(img):
    # 显示图片
    npimg = img.numpy()
    # 最近插值法(Nearest Neighbor Interpolation)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.pause(1)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def show_weights_hist(data):
    plt.hist(data, bins=100, normed=1, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.title("D weights")
    plt.show()


if __name__ == '__main__':
    train_dataset = load_car_data()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    show(torchvision.utils.make_grid(denorm(next(iter(trainloader))[0]), nrow=8))
