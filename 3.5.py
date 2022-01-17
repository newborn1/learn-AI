import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
torch.manual_seed(1)#由于产生随机种子

if __name__=='__main__':
    BATCH_SIZE = 8

    x = torch.linspace(0,100,10)
    y = torch.linspace(0,200,10)

    torch_dataset = Data.TensorDataset(x,y)#打包数据

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        timeout=3,#设置超时时间，如果数据超过这个时间还没读完就报错
        num_workers=2,#多线程程序要放在主函数中训练
    )

    for epoch in range(3):#训练次数
        for step, (batch_x,batch_y) in enumerate(loader):
            print('Epoch:{0} | Step:{1} | batch x:{2} | batch y:{3}'.format(
                epoch,step,batch_x,batch_y))