import os
from pickletools import optimize
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"#有多个gpu才有用
os.environ["CUDA_VISIBLE_DEVICES"]='0'#有多个gpu才有用,选哪一个

import time
import torchvision
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)#打印出gpu才进行运行

torch.manual_seed(1)    # reproducible


# Hyper Parameters,以后可以用配置文件来进行模块化
KERNEL_SIZE_LAYER1 = 5
POOL_SIZE_LAYER1 = 2
KERNEL_SIZE_LAYER2 = 5
POOL_SIZE_LAYER2 = 2
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False



#定义卷积神经网络
class CNNNet(nn.modules):
    def __init__(self) -> None:#把需要参数训练的才叫做一层神经网络，池化层和激励层不需要，故而不单独放为一层
        super(CNNNet).__init__()
        self.convLayer1 = nn.Sequential(#输入(1,28,28) 输出(16,14,14)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=KERNEL_SIZE_LAYER1,
                stride=1,
                padding=(KERNEL_SIZE_LAYER1 - 1)//2,#向下取整
            ),#卷积层
            nn.ReLU(),#激励层
            nn.MaxPool2d(
                kernel_size=POOL_SIZE_LAYER1,
                stride = 1,
            )#最大池化层
        )

        self.convLayer2 = nn.Sequential(#输入(16,14,14) 输出(32,7,7)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=KERNEL_SIZE_LAYER2,
                stride=1,
                padding=(KERNEL_SIZE_LAYER2-1)//2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=POOL_SIZE_LAYER2,
                stride=1
            )
        )

        self.denseLayer3 = nn.Sequential(#输入(32*7*7) 输出(10)——检测10为数字则输出一定要是10
            nn.Linear(32*7*7,10),#全连接层
            nn.Softmax(10)#激励层
        )


    def forward(self,x):
        x = self.convLayer1(x)#注意这里的用法
        x = self.convLayer2(x)

        x = x.view(x.size(0),-1)#要将其展开        
        output = self.denseLayer3(x)


        return output

#定义main函数
def main():
    #建立数据:Mnist 手写数字
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    # 保存或者提取位置,即图片地址
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
    )
    #test
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

    train_data,test_data = train_data.to(device),test_data.to(device)
    print(train_data.device,test_data.device)


    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(#这里的dataset不是很理解，为什么没有标签
        dataset=train_data,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    # 为了节约时间, 我们测试时只测试前2000个
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    cnn = CNNNet()
    cnn.to(device)
    print(cnn)

    optimize = torch.optim.Adam(cnn.parameters(),lr=LR,betas=(0.9,0.99))
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    #开始训练模型
    for epoch in range(EPOCH):
        for step,(batch_x,batch_y) in enumerate(train_loader):
            output = cnn(batch_x)
            loss = loss_func(output,batch_y)

            optimize.zero_grad()
            loss.backward()
            optimize.step()

            print('第{}次的第{}批的损失是{}'.format(epoch,step,loss))



    #测试模型
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')


if __name__=='__main__':
    main()
