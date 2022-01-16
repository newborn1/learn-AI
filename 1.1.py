from dataclasses import dataclass
from operator import matmul
import torch
import numpy as np

data = [[1,2],[3,4]]
#dataArray = np.array(data,dtype=list)#这样不能转换为dataTensor,错的
dataArray = np.array(data)#一般情况不要加dtype
dataTensor = torch.from_numpy(dataArray)
dataToArray = dataTensor.numpy()


print(
    "\nnumpy:",dataArray,
    "\ntorch:",dataTensor,
    "\ntorch to numpy:",dataToArray,
    '\ntorch.mm:',torch.mm(dataTensor,dataTensor),
    '\nnumpy.matmul:',np.matmul(dataArray,dataArray)
)