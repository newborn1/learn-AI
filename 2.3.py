import os#要加上这两句且必须放在最前面，因为有库的版本有冲突问题：numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-10,10,400)
x = Variable(x)
fig,ax = plt.subplots(2,2,figsize=(10,6))

y_relu = F.relu(x).data.numpy()
ax[0,0].plot(x,y_relu,c='red',label='relu')
ax[0,0].legend(loc='best')

y_sigmoid = F.sigmoid(x).data.numpy()
ax[0,1].plot(x,y_sigmoid,c='green',label='sigmoid')
ax[0,1].legend(loc='best')

y_tanh = F.tanh(x)
ax[1,0].plot(x,y_tanh,c='black',label='tanh')
ax[1,0].legend(loc='best')

y_softplus = F.softplus(x)
ax[1,1].plot(x,y_softplus,c='yellow',label='softplus')
ax[1,1].legend(loc='best')

plt.show()