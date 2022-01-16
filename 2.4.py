import os#要加上这两句且必须放在最前面，因为有库的版本有冲突问题：numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

matplotlib.rcParams['font.family'] = 'Kaiti'
matplotlib.rcParams['axes.unicode_minus']=False #用来正常显示负号  

#建立数据集
x = torch.linspace(-1,1,100)
x = Variable(x)
y = pow(x,2) + 0.1*torch.rand(x.size())#增加噪声,注意size要加括号

fig,ax = plt.subplots(1,2,figsize=(10,6))
ax[0].scatter(x,y,c='red')
ax[0].plot(x,pow(x,2),c='blue',label='y=x*x')
ax[0].set_title('这是样本')

ax[0].legend(loc='best')

plt.show()


#构建神经网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()

        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forword(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net =Net(n_feature=1,n_hidden=10,n_output=1)

print(net)

#训练网络
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
loss_func = torch.nn.MSELoss()

for i in range(100):
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%5 == 0:
        plt.cla()
        plt.pause(0.1) # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
