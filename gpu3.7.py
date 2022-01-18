import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
#run time : 74.8222161s
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"#有多个gpu才有用
os.environ["CUDA_VISIBLE_DEVICES"]='0'#有多个gpu才有用,选哪一个

# plt.switch_backend('agg')#BUG: Backend TkAgg is interactive backend. Turning interactive mode on.

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.current_device())

import sys
print(sys.version)

# 声明设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)#打印出gpu才进行运行

torch.manual_seed(1)    # reproducible

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

    nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]#将模型转移到GPU中，实际上是将参数转移到GPU
    for net in nets:
        net.to(device)
    
    print(next(nets[1].parameters()).device)#查看模型所放位置在哪

    #定义四个不同的优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Mom = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMS = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

    opts = [opt_SGD,opt_Mom,opt_RMS,opt_Adam]

    losses_his = [[],[],[],[]] #记录训练时不同神经网络的 loss

    #建立数据集,并用转化为TensorSet,同时打包再DataLoader中处理
    x = torch.linspace(-1,1,1000).reshape(-1,1)#设为0-100则要规范化
    y = x*x + 0.1*torch.normal(torch.zeros(*x.size()))#啥意思？？？

    x,y = x.to(device=device),y.to(device=device)#gpu
    print(x.device)#查看数据所在的设备

    dataset = Data.TensorDataset(x,y)
    BATCH_SIZE = 32
    TIMES = 12 #训练次数
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    # loader = loader.to(device)

    loss_func = torch.nn.MSELoss()

    # for times in range(TIMES):
    # #这里涉及进程问题：Exception has occurred: RuntimeError cuda runtime error (801) : operation not supported at ..\torch/csrc/generic/StorageSharing.cpp:249。要么换成单线程要么在子线程才将模型放在gpu里
    #     for step,(batch_x,batch_y) in enumerate(loader):
    #         # print(batch_x.device,batch_y.device)
    #         for net,opt,l_his in zip(nets,opts,losses_his):

    #             # batch_x, batch_y= batch_x.to(device),batch_y.to(device)#gpu,不用加，GPU计算后的数据仍然存放在GPU里？
                
    #             output = net(batch_x)
    #             output = output.to(device)
    #             loss = loss_func(output,batch_y)
    #             opt.zero_grad()
    #             loss.backward()
    #             opt.step()
    #             l_his.append(loss.data)


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
    #这里涉及进程问题：Exception has occurred: RuntimeError cuda runtime error (801) : operation not supported at ..\torch/csrc/generic/StorageSharing.cpp:249。要么换成单线程要么在子线程才将模型放在gpu里
        for step,(batch_x,batch_y) in enumerate(loader):
            # print(batch_x.device,batch_y.device)
            ax.cla()#四条画好了才除去
            for net,opt,l_his,color in zip(nets,opts,losses_his,colors):

                # batch_x, batch_y= batch_x.to(device),batch_y.to(device)#gpu,不用加，GPU计算后的数据仍然存放在GPU里？
                
                output = net(batch_x)
                output = output.to(device)#important
                loss = loss_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss = loss.cpu()
                # print(loss.data)
                l_his.append(loss)

                ax0 = range(len(l_his))
                # print(l_his[0].device
                # l_his[i] = l_his[i].data.cpu().numpy()#不是data()，loss.data放GPU上,要转化为CPU供plot用，要转化为 numpy()，否则会出现:Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
                ax.plot(
                    ax0,
                    torch.tensor(l_his).cpu().data.numpy(),#重要
                    c=color,
                    # label = label#在这里设无效
                )#本来就放在CPU上，不需要转,并且只有tensor类型的数据才能和device有关系

            # ax.set_xlim(0,int(TIMES*200/BATCH_SIZE))
            ax.set_xlabel('Times')
            ax.set_ylim(0.000,0.200)
            ax.set_ylabel('Loss')

            ax.legend(labels,loc='best')#要在这里设
            plt.pause(0.1)
    
    print('运行时间是 %s s'%(time.perf_counter()-t))
    plt.show()
