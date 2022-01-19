"""
    这是一个关于Q-learning的代码
"""

import math
import os
import string#要加上这两句且必须放在最前面，因为有库的版本有冲突问题：numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.autograd.functional as F

import time
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(2)

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000      # 记忆库大小
ACTION = ['left','right']
ACTIONS_N = len(ACTION)

class QLearning(object):
    def __init__(self) -> None:
        super(QLearning).__init__()
        self.batchSize = BATCH_SIZE
        self.lr = LR                  
        self.epsilon = EPSILON       
        self.gamma = GAMMA             
        self.targetReplaceIter = TARGET_REPLACE_ITER
        self.memoryCapacity =  MEMORY_CAPACITY
        self.action = ACTION
        self.actionNum = ACTIONS_N
        self.memory = pd.DataFrame(#建立表格储蓄数据信息
            data=torch.zeros(self.memoryCapacity,self.actionNum),
            index=None,
            columns=self.action,
            # dtype=torch.float64
        )

    def choose_action(self,s):#根据当前状态s选择合适的行为,后面可以根据深度学习来优化
        suitable_action = self.memory.loc[s,:]
        if np.random.uniform() < self.epsilon:
            argmax = suitable_action[suitable_action == suitable_action.max()].index#获得最大的列所在的索引
            argmax_action = np.random.choice(argmax)#随便选择一个最大值对应的action
        else:
            argmax = np.random.randint(0, self.actionNum)
            argmax_action = self.action[argmax]

        return argmax_action

    def store_transition(self,s,a,r,s_):
        a_ = self.choose_action(s_)
        self.transition = pd.DataFrame(
            data = np.array([s,a,r,s_,a_]).reshape(1,5),#Shape of passed values is (5, 1), indices imply (5, 5)
            index=None,
            columns=['当前状态','当前选择行为','奖励','下一个状态','下一状态选择的行为'],
            # dtype=np.dtype([
            #     ('s',np.float64),
            #     ('a',np.str0),
            #     ('r',np.float64),
            #     ('s_',np.float64),
            #     ('a_',np.str0)
            # ])
        )

    def learn(self) -> None:
        (s,a,r,s_,a_) = self.transition.loc[0,:]
        (s,r,s_) = np.float64((s,r,s_))
        #更新状态，维护状态表
        self.memory.loc[s,a] += self.lr*(r + self.gamma*self.memory.loc[s_,a_] - self.memory.loc[s,a])


def main():
    qLearnTable = QLearning()
    print(qLearnTable.memory.loc[:,:])
    s = np.random.randint(0,MEMORY_CAPACITY)
    ecpo = 1
    while ecpo<1000 :
        ecpo += 1
        a = qLearnTable.choose_action(s)
        # s_ = 2 * (1 if a == 'right' else 2)
        s_ = np.random.randint(0,MEMORY_CAPACITY)#这里需要有游戏环境
        r = math.sqrt(s_**2 + s_**2)
        qLearnTable.store_transition(s,a,r,s_)
        qLearnTable.learn()
        s = s_
    
    print(qLearnTable.memory.loc[:,:])

if __name__ == '__main__':
    main()
