# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:27:09 2021

@author: 54398
"""

import numpy as np
import torch
import torch.nn as nn

import scipy.sparse as sp
# from tool.qmath import cosine
import torch.autograd as Variable

import torch.nn.functional as F


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

import time

from train.gpu_memory_log import gpu_memory_log


    
    
    
class RGCN(nn.Module):
    
    def __init__(self,  num_users, num_items, embed_dim, n_layers,
                 reg_weight, dataset):
        super(RGCN,self).__init__()
        

        self.n_users = num_users
        self.n_items = num_items

        self.latent_dim = embed_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        
        self.embed_dim = embed_dim
        
     
        self.listed_data = []
   
        self.multi_adj = dataset.multi_adj_new
        self.multi_social_adj = dataset.multi_social_new
        
        self.hide_dim = embed_dim
        self.ratingClass = 5
        self.ratingSocial = 5
        
        self.user_embedding = nn.Embedding(self.n_users*self.ratingSocial, self.latent_dim)#.to(self.device)
        self.item_embedding = nn.Embedding(self.n_items*5, self.latent_dim)#.to(self.device)
        self.w = nn.Embedding(self.n_items, self.latent_dim)
        
        nn.init.normal_(self.user_embedding.weight,mean=0.,std=0.1)
        nn.init.normal_(self.item_embedding.weight,mean=0.,std=0.1)
        nn.init.normal_(self.w.weight,mean=0.,std=0.1)
        
        
        self.w_r1 = nn.Linear(self.latent_dim * 2, self.latent_dim)#.to(self.device)
        self.w_r2 = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.u_r1 = nn.Linear(self.latent_dim * 2, self.latent_dim)#.to(self.device)
        self.u_r2 = nn.Linear(self.latent_dim, self.latent_dim)
        
  
        self.criterion = nn.MSELoss()
        
        self.layer = [64,64]

        
        
        initializer = nn.init.xavier_uniform_
        
        self.act = torch.nn.PReLU()
        self.weight_dict = nn.ParameterDict()
        self.social_weight_dict = nn.ParameterDict()
        for k in range(1, self.n_layers):
            if k == 0:
                self.weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(torch.empty(self.hide_dim, self.layer[k])))})
                self.weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(torch.empty(self.hide_dim, self.layer[k])))})
            else:
                self.weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(torch.empty(self.layer[k-1], self.layer[k])))})
                self.weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(torch.empty(self.layer[k-1], self.layer[k])))})
                
        for k in range(1, self.n_layers):
            if k == 0:
                self.social_weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(torch.empty(self.hide_dim, self.layer[k])))})
                # self.social_weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(torch.empty(self.hide_dim, self.layer[k])))})
            else:
                self.social_weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(torch.empty(self.layer[k-1], self.layer[k])))})
                # self.social_weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(torch.empty(self.layer[k-1], self.layer[k])))})
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        # self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
    
    def getSocialEmbedding(self):
        
        

        
        user_embeddings = self.user_embedding.weight
        # print('multi_social_adj',self.multi_social_adj.shape)
        # print('user_embeddings',user_embeddings.shape)
        
        embeddings = self.act(torch.spmm(self.multi_social_adj, user_embeddings))
        all_user_embeddings = [embeddings]
        
        for k in range(1,self.n_layers):
            ego_embeddings = torch.mm(all_user_embeddings[-1],self.social_weight_dict['user_w%d' %k]
                                      )
        
            
            # ego_embeddings = torch.cat((tmp_user_embed,tmp_item_embed),dim=0)
            embeddings = self.act(torch.spmm(self.multi_social_adj,ego_embeddings))
            
            all_user_embeddings += [embeddings]
            # all_item_embeddings += [embeddings[self.n_users:]]
            
            
        # user_embedding = torch.cat(all_user_embeddings,1) # 
        # print('user_embedding',user_embedding.shape) # user_embedding torch.Size([2964, 128]) 
        user_multi_embedding = torch.cat(all_user_embeddings,1)
        # print('item_embedding',item_embedding.shape) # item_embedding torch.Size([198475, 128])
        
        user_multi_embedding = user_multi_embedding.view(-1, self.ratingSocial, 128) # 64+64
            #mean or attention
        user_social_embeddings = torch.div(torch.sum(user_multi_embedding, dim=1), 4)
        # print('item_embedding',item_embedding.shape)
        
        
        
        return user_social_embeddings
            
            
            
        
    
    def get_ego_embeddings(self,social_adj):
        
        user_embeddings = self.getSocialEmbedding(social_adj)
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings,item_embeddings], dim=0)
        
        return ego_embeddings

    
    def forward(self):
        
        user_embeddings = self.w_r1(self.getSocialEmbedding())
        item_embeddings = self.item_embedding.weight
        
        ego_embeddings= torch.cat([user_embeddings,item_embeddings], dim=0)
        
        embeddings = self.act(torch.spmm(self.multi_adj, ego_embeddings))
        
        
        # original embedding
        all_user_embeddings = [embeddings[:self.n_users]]
        all_item_embeddings = [embeddings[self.n_users:]]
        
        
        for k in range(1,self.n_layers):
            tmp_user_embed = torch.mm(all_user_embeddings[-1],self.weight_dict['user_w%d' %k]
                                      )
            tmp_item_embed = torch.mm(all_item_embeddings[-1],self.weight_dict['item_w%d' %k])
            
            ego_embeddings = torch.cat((tmp_user_embed,tmp_item_embed),dim=0)
            embeddings = self.act(torch.spmm(self.multi_adj,ego_embeddings))
            
            all_user_embeddings += [embeddings[:self.n_users]]
            all_item_embeddings += [embeddings[self.n_users:]]
            
            
        user_embedding = torch.cat(all_user_embeddings,1) # 
        # print('user_embedding',user_embedding.shape) # user_embedding torch.Size([2964, 128]) 
        item_multi_embedding = torch.cat(all_item_embeddings,1)
        # print('item_embedding',item_embedding.shape) # item_embedding torch.Size([198475, 128])
        
        item_muliti_embed = item_multi_embedding.view(-1, self.ratingClass, 128) # 64+64
            #mean or attention
        item_embedding = torch.div(torch.sum(item_muliti_embed, dim=1), 5)
        # print('item_embedding',item_embedding.shape)
        
        
        return user_embedding, item_embedding

        
    
        
    def predict(self,nodes_u,nodes_v):
        
        # norm_social_adj_matrix = self.generator.forward()
        # norm_social_adj_matrix = social_adj#self.get_norm_social_adj_mat().to(self.device)
        
        # user_all_embeddings, item_all_embeddings = self.forward(social_adj)
        # print('7',social_adj.shape)
        embeds_u, embeds_v = self.forward()
        
        embeds_u = self.u_r2(self.u_r1(embeds_u[nodes_u]))
        embeds_v = self.u_r2(self.u_r1(embeds_v[nodes_v]))
        
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        # print(x_u.shape)
        # print(x_v.shape)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()
    '''
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embedding = user_all_embeddings[nodes_u]
        i_embedding = item_all_embeddings[nodes_v]
        
        scores = torch.mul(u_embedding, i_embedding).sum(dim=1)
        
        return scores
    '''
    
    def loss(self,nodes_u,nodes_v,labels_list):
        
     

        
        # user_all_embeddings, item_all_embeddings = self.forward()


        # u_embedding = user_all_embeddings[nodes_u]
        # i_embedding = item_all_embeddings[nodes_v]
        
        scores = self.predict(nodes_u,nodes_v)#.to(self.device)
        
        
 
        loss = self.criterion(scores,labels_list)

        # print('10')
        return loss
    
    def rating_loss(self,nodes_u,nodes_v,labels_list, social_adj):
        norm_social_adj_matrix = social_adj
        user_all_embeddings, item_all_embeddings = self.forward(norm_social_adj_matrix)
        u_embedding = user_all_embeddings[nodes_u]
        i_embedding = item_all_embeddings[nodes_v]
        # print(u_embedding.device)
        
        scores = torch.mul(u_embedding, i_embedding).sum(dim=1)#.to(self.device)
        
        scores = self.predict(nodes_u, nodes_v,social_adj)
        s_loss = self.criterion(labels_list, scores)
        
        return s_loss
        
        
    