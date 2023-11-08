# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:10:44 2020

@author: shuaili
"""
import torch
import torch.nn as nn
import numpy as np


# m = nn.Bilinear(20, 30, 40)
# input1 = torch.randn(128, 20)
# input2 = torch.randn(128, 30)
# output = m(input1, input2)
# print(output.size())

# # torch.Size([128, 40])

# print('learn nn.Bilinear')
# m = nn.Bilinear(20, 30, 40)
# input1 = torch.randn(128, 20)
# input2 = torch.randn(128, 30)
# output = m(input1, input2)
# print(output.size())
# arr_output = output.data.cpu().numpy()
 
# weight = m.weight.data.cpu().numpy()
# bias = m.bias.data.cpu().numpy()
# x1 = input1.data.cpu().numpy()
# x2 = input2.data.cpu().numpy()
# print(x1.shape,weight.shape,x2.shape,bias.shape)
# y = np.zeros((x1.shape[0],weight.shape[0]))
# for k in range(weight.shape[0]):
#     buff = np.dot(x1, weight[k])
#     buff = buff * x2
#     buff = np.sum(buff,axis=1)
#     y[:,k] = buff
# y += bias
# dif = y - arr_output
# print(np.mean(np.abs(dif.flatten())))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
u2e = nn.Embedding(64, 64).cuda()



# u = [1,2,3]
# c = []

# for i in u:
#     print(i)