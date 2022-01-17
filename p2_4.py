import os#要加上这两句且必须放在最前面，因为有库的版本有冲突问题：numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

matplotlib.rcParams['font.family'] = 'Kaiti'
matplotlib.rcParams['axes.unicode_minus']=False #用来正常显示负号  


#构建神经网络
class Net(torch.nn.Module):#一般值存放数据发生变化的模型参数
    
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()

        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=10,n_output=1)

print(net)

if __name__=='__main__':
    #建立数据集
    x = torch.linspace(-1,1,100).reshape(-1,1)#一维，即一个特征
    # x = Variable(x)
    y = pow(x,2) + 0.1*torch.rand(x.size())#增加噪声,注意size要加括号

    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].scatter(x,y,c='red')
    ax[0].plot(x,pow(x,2),c='blue',label='y=x*x')
    ax[0].set_title('这是样本')

    ax[0].legend(loc='best')

    # plt.show()#这个不能加上去，否则下面的不会出现




    #训练网络
    optimizer = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
    loss_func = torch.nn.MSELoss()#平方损失函数

    for i in range(200):#600为学习次数
        prediction = net(x)

        loss = loss_func(prediction,y)

        optimizer.zero_grad()# 清空上一步的残余更新参数值
        loss.backward()# 误差反向传播, 计算参数更新值
        optimizer.step()# 将参数更新值施加到 net 的 parameters 上

        if i%5 == 0:
            #展示学习的过程
            ax[1].cla()
            ax[1].scatter(x.data.numpy(), y)
            ax[1].plot(x,pow(x,2),c='blue',label='y=x*x')
            ax[1].plot(x.data, prediction.data, 'r-', lw=5)
            ax[1].text(0.5, 0, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color':  'red'})#添加注释
            plt.pause(0.1)

    torch.save(net.state_dict(),'./莫烦python/net_params/2.4net_params.pkl')#相对地址别写错

    plt.show()#加上不然会消失