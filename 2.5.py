import os#要加上这两句且必须放在最前面，因为有库的版本有冲突问题：numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("./莫烦python/_2.4.py")

#先new一个一模一样的网络
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from p2_4 import Net

net = Net(1,10,1)#参数要写好

#载入参数
net.load_state_dict(torch.load('./莫烦python/net_params/2.4net_params.pkl'),strict=True)

print("引入后的网络为：",net)
