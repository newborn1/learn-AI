import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#run time 484.2945112s
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

class TextNet(nn.Module):
    def __init__(self,in_features=1,n_hidden=10,out_features=1) -> None:
        super(TextNet,self).__init__()
        self.hidden = nn.Linear(in_features,n_hidden)
        self.output = nn.Linear(n_hidden,out_features)

    def forward(self,x):
        x = self.hidden(x)
        x = self.output(x)
        
        return x


if __name__ == '__main__':
    t = time.perf_counter()#查看运行时间
    LR = 0.2
    #定义四个相同的网络结构
    net_SGD = TextNet()
    net_Momentum = TextNet()
    net_RMSprop = TextNet()
    net_Adam = TextNet()

    nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

    #定义四个不同的优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Mom = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMS = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

    opts = [opt_SGD,opt_Mom,opt_RMS,opt_Adam]

    losses_his = [[],[],[],[]] #记录训练时不同神经网络的 loss

    #建立数据集,并用转化为TensorSet,同时打包再DataLoader中处理
    x = torch.linspace(-1,1,200).reshape(-1,1)
    y = x*x + 0.1*torch.normal(torch.zeros(*x.size()))#啥意思？？？

    dataset = Data.TensorDataset(x,y)
    BATCH_SIZE = 50
    TIMES = 100#训练次数
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    loss_func = torch.nn.MSELoss()

    #这是边训练边画图
    fig,ax = plt.subplots(1,1,figsize=(10,6))

    ax0 = range(int(TIMES*200/BATCH_SIZE))
    colors = ['r','g','blue','pink']
    labels = ['SGD','Moemenutm','RMSprop','Adam']

    ax.set_xlim(0,int(TIMES*200/BATCH_SIZE))
    ax.set_xlabel('Times')
    ax.set_ylim(0.00,0.25)
    ax.set_ylabel('Loss')

    for times in range(TIMES):
        for step,(batch_x,batch_y) in enumerate(loader):
            ax.cla()#四条画好了才除去
            for net,opt,l_his,color in zip(nets,opts,losses_his,colors):
                output = net(batch_x)
                loss = loss_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

                ax0 = range(len(l_his))
                ax.plot(ax0,l_his,c=color)

            # ax.set_xlim(0,int(TIMES*200/BATCH_SIZE))
            ax.set_xlabel('Times')
            ax.set_ylim(0.06,0.60)
            ax.set_ylabel('Loss')

            ax.legend(labels,loc='best')
            plt.pause(0.1)
    
    print('运行时间是 %s s'%(time.perf_counter()-t))
    plt.show()

    
"""
    #画图训练分开

    for times in range(TIMES):
        for step,(batch_x,batch_y) in enumerate(loader):
            for net,opt,l_his in zip(nets,opts,losses_his):
                output = net(batch_x)
                loss = loss_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
    fig,ax = plt.subplots(1,1,figsize=(10,6))

    ax0 = range(TIMES*200/BATCH_SIZE)#200是总体样本总数
    colors = ['r','g','blue','pink']
    for loss_his,color in zip(losses_his,colors):#要用zip否则会报错
        ax.plot(ax0,loss_his,c=color)

    ax.set_xlim(0,TIMES*200/BATCH_SIZE)
    ax.set_xlabel('Times')
    ax.set_ylim(0.00,0.25)
    ax.set_ylabel('Loss')
    
    ax.legend('best')
    plt.show()
"""