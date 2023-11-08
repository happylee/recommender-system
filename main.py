# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:25:30 2021

@author: 54398
"""


import numpy as np
import torch
import torch.nn as nn

import scipy.sparse as sp
# from tool.qmath import cosine
import torch.autograd as Variable

import torch.nn.functional as F

import dataload as dataload
from torch.nn import init
# from torch.autograd import Variable
import pickle
import time
import random
from collections import defaultdict

# torch.cuda.set_device(1)
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import numpy as np

import time



from RGCN import RGCN



def train(model, train_loader, optimizer, epoch, best_rmse, best_mae, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
  
    for i, data in enumerate(train_loader, 0):
    
        batch_nodes_u, batch_nodes_v, labels_list = data
        batch_nodes_u, batch_nodes_v, labels_list = batch_nodes_u.to(device), batch_nodes_v.to(device),labels_list.to(device)
        
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u,batch_nodes_v,labels_list)
        

        
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 0:
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch, i, running_loss / 100, best_rmse, best_mae))
            # running_loss = 0.0
    elapsed_time = time.time()-start_time
    print('one train time',time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    return 0


def test(model, test_loader, device):
    model.eval()

    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.predict(test_u, test_v)
            # print(val_output)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
            
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def main():
    #Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--dataset_path', default='G:/recommender/pytorch/owncode/nanshou/datasets_pre/toy', help='input batch size for training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()
    
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    

    embed_dim = args.embed_dim
    dataset = dataload.BasicDataset()
    train_u,train_v,train_r,valid_u,valid_v,valid_r,test_u, test_v, test_r, user_count, item_count, multi_social,multi_adj_new = dataset.getInfo() 
    
    

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                               torch.FloatTensor(train_r))
    # print(social_mat)
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                              torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=0)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,num_workers=0)
   
    
    num_users = user_count
    num_items = item_count 
    print(num_users)
    print(num_items)


    # model
    # gpu_memory_log
    model = RGCN(num_users, num_items, embed_dim, 2, 0.1, dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # gpu_memory_log()
    

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    test_best_rmse = 999.0
    test_best_mae = 999.0

    for epoch in range(100):#(1, args.epochs + 1):
        print('train',epoch)

        train(model,train_loader, optimizer, epoch, best_rmse, best_mae, device)
        print('test',epoch)
        expected_rmse, mae = test(model, valid_loader, device)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            test_expected_rmse, test_mae = test(model, test_loader, device)
            if test_best_rmse > test_expected_rmse:
                test_best_rmse = test_expected_rmse
                test_best_mae = test_mae
                print("test_rmse: %.4f, test_mae:%.4f " % (expected_rmse, mae))
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        # if epoch % 5 == 0:
        #     expected_rmse1, mae1 = test(lightGCN, device, test_loader)
        #     print("rmse: %.4f, mae:%.4f " % (expected_rmse1, mae1))

        if endure_count > 10:
            break


if __name__ == "__main__":
    main()