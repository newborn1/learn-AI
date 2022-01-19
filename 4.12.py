from email.generator import Generator
from this import d
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

#超参数
"""
https://mofanpy.com/tutorials/machine-learning/torch/GAN/#%E8%AE%AD%E7%BB%83
https://www.cnblogs.com/js2hou/p/13923089.html
新手画家 (Generator) 在作画的时候需要有一些灵感 (random noise), 我们这些灵感的个数定义为 N_IDEAS.
而一幅画需要有一些规格, 我们将这幅画的画笔数定义一下, N_COMPONENTS 就是一条一元二次曲线(这幅画画)上的点个数. 
为了进行批训练, 我们将一整批话的点都规定一下(PAINT_POINTS).
"""
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

#著名画家的画：即需要人为给画家的数据，以供Discriminator进行学习
def artist_works() ->torch.float64:
    a = np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    paintings = a * np.power(PAINT_POINTS,2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    paintings = paintings.to(device)

    return paintings

#神经网络
Generator = nn.Sequential(
    nn.Linear(N_IDEAS,128),
    nn.ReLU(),
    nn.Linear(128,ART_COMPONENTS)
)
Generator.to(device=device)

Discriminator = nn.Sequential(
    nn.Linear(ART_COMPONENTS,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid()
)
Discriminator.to(device=device)

opt_G = torch.optim.Adam(
    Generator.parameters(),
    lr=LR_G,
    betas=(0.9,0.99)
)

opt_D = torch.optim.Adam(
    Discriminator.parameters(),
    lr=LR_D,
    betas=(0.9,0.99)
)

#训练
for step in range(10000):
    artist_paintings = artist_works()
    # artist_paintings = artist_paintings.to(device)
    G_ideas = torch.randn(BATCH_SIZE,N_IDEAS)
    G_ideas = G_ideas.to(device)
    G_paintings = Generator(G_ideas)

    prob_artist0 = Discriminator(artist_paintings)
    prob_artist1 = Discriminator(G_paintings)

    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)# retain_graph 这个参数是为了再次使用计算图纸

    opt_G.zero_grad()
    G_loss.backward()
    opt_D.step()
    opt_G.step()
    
