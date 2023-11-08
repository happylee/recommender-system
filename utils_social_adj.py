# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:18:29 2021

@author: 54398
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:08:14 2020

@author: 54398
"""



from collections import defaultdict

import torch
import pickle
import pandas as pd

from scipy.io import loadmat

import argparse

from tqdm import tqdm

import numpy as np
import random

import scipy.sparse as sp


path = 'G:/recommender/pytorch/owncode/nanshou/datasets_pre/toy'
train_path = path + 'train.txt'
test_path = path + 'test.txt'
rating_path = path + 'rating.txt'

dir_data = 'G:/recommender/pytorch/GraphRec_test/origin_code/datasets_pre/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Epinions', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.2, help='the proportion of data used for test')
args = parser.parse_args()

# =============================================================================
# Epinions users: 49290 items:139739 social connections 
# =============================================================================
workdir = '/data1/lss/project/RGCN/datasets_pre/'

if args.dataset == 'Ciao':
# 	click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
# 	trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
    click_f = np.loadtxt(workdir + 'Ciao/rating.txt', dtype = np.int32)
    trust_f = np.loadtxt(workdir + 'Ciao/trust.txt', dtype = np.int32)
    train_f = np.loadtxt(workdir+'Ciao/train.txt', dtype = np.int32)
    test_f = np.loadtxt(workdir+'Ciao/test.txt', dtype = np.int32)
    # train_file = workdir+'Ciao/train.txt'
    # test_file = workdir+'Ciao/test.txt'
elif args.dataset == 'toy':
 	click_f = np.loadtxt(workdir+'Epinions/rating1.txt', dtype = np.int32)
 	trust_f = np.loadtxt(workdir+'Epinions/trust1.txt', dtype = np.int32)
# 	click_f = loadmat(workdir + 'Epinions/rating.mat')['rating']
# train_f = np.loadtxt(workdir+'Epinions/train.txt', dtype = np.int32)
    # test_f = np.loadtxt(workdir+'Epinions/test.txt', dtype = np.int32)
# 	trust_f = loadmat(workdir + 'Epinions/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
    click_f = np.loadtxt(workdir+'Epinions/rating.txt', dtype = np.int32)
    trust_f = np.loadtxt(workdir+'Epinions/social.txt', dtype = np.int32)
    train_f = np.loadtxt(workdir+'Epinions/train.txt', dtype = np.int32)
    test_f = np.loadtxt(workdir+'Epinions/test.txt', dtype = np.int32)
elif args.dataset == 'gowalla':
    train_file = workdir+'gowalla/train.txt'
    test_file = workdir+'gowalla/test.txt'
elif args.dataset == 'douban':
    click_f = np.loadtxt(workdir+'douban/rating.txt', dtype = np.int32)
    trust_f = np.loadtxt(workdir+'douban/trust.txt', dtype = np.int32)
    train_f = np.loadtxt(workdir+'douban/train.txt', dtype = np.int32)
    test_f = np.loadtxt(workdir+'douban/test.txt', dtype = np.int32)

train_list = []
test_list = []
trust_list = []

m_item = 0
n_user = 0
rate_count = 0
for s in train_f:
    uid = s[0]
    iid = s[1]
    if args.dataset == 'toy':
        label = s[2]
    else:
        label = s[2]
        
    if uid > n_user :
        n_user = uid
    if iid > m_item :
        m_item = iid
    if label > rate_count :
        rate_count = label
    train_list.append([uid, iid, label])
    
for s in test_f:
    uid = s[0]
    iid = s[1]
    if args.dataset == 'toy':
        label = s[2]
    else:
        label = s[2]
        
    if uid > n_user :
        n_user = uid
    if iid > m_item :
        m_item = iid
    if label > rate_count :
        rate_count = label
    test_list.append([uid, iid, label])
    
for s in trust_f:
    uid = s[0]
    fid = s[1]
        
    if uid > n_user :
        n_user = uid
    if fid > n_user :
        n_user = fid
    if label > rate_count :
        rate_count = label
    trust_list.append([uid, fid])



            
            

user_count = n_user + 1
item_count = m_item + 1
print(user_count)

train_u = []
train_v = []
train_r = []
test_u = []
test_v = []
test_r = []

trust_uid = []
trust_fid = []


# for i,data in enumerate(train_f):
#     train_u.append(data[0])
#     train_v.append(data[1])
#     train_r.append(data[2])
    
# for i,data in enumerate(test_f):
#     test_u.append(data[0])
#     test_v.append(data[1])
#     test_r.append(data[2])
    
for i,data in enumerate(trust_f):
    trust_uid.append(data[0])
    trust_fid.append(data[1])
    
# adj_list = sp.coo_matrix((train_r,(train_u,train_v)),shape=(user_count,item_count),dtype=np.float32)



# users_D = np.array(adj_list.sum(axis=1)).squeeze()



#        print(self.users_D.shape)
# users_D[users_D == 0.] = 1
# items_D = np.array(adj_list.sum(axis=0)).squeeze()
# items_D[items_D == 0.] = 1.
# print(adj_list)


# =============================================================================
# social_list = sp.coo_matrix((np.ones(len(trust_uid)),(trust_uid,trust_fid)),shape=(user_count,user_count),dtype=np.float32)
# print('social',social_list.todense())
# =============================================================================

trust_df = pd.DataFrame(trust_f, columns = ['uid', 'fid'])




rating_social_list = sp.coo_matrix(np.zeros((user_count,user_count),dtype=np.float32))
rating_social_list = rating_social_list.todense()
# print(rating_social_list)
print(rating_social_list[2000,2000])

trust_list1 = []
for u in tqdm(range(user_count + 1)):
    hist = trust_df[trust_df['uid'] == u]
    f_users = hist['fid'].tolist()
    # i_labels = hist['label'].tolist()
    if len(f_users) > 0:
        for f in f_users:
            hist_f = trust_df[trust_df['uid'] == f]
            f_f_users = hist_f['fid'].tolist()
            num = len(set(f_users) & set(f_f_users))
            rating_social_list[u,f] = num
            trust_list1.append((u,f,num))


ratingClass = np.unique(rating_social_list.data).size #42

nums = [0,0,0,0,0,0,0,0,0,0]

rows = rating_social_list.shape[0]
cols = rating_social_list.shape[1]
#rating_social_list1 = rating_social_list
rating_social_list1 = sp.coo_matrix(np.zeros((user_count,user_count),dtype=np.float32))
rating_social_list1 = rating_social_list1.todense()



    
social_df = pd.DataFrame(trust_list1, columns = ['uid', 'fid','label'])

for i in tqdm(range(user_count + 1)):
    hist = social_df[social_df['uid']==i]
    f_users = hist['fid'].tolist()
    for j in f_users:
        if rating_social_list[i,j] < 5 and rating_social_list[i,j] > 0:
            nums[0] += 1
            rating_social_list1[i,j] = 1
        elif rating_social_list[i,j] < 10 and rating_social_list[i,j] >= 5:
            nums[1] += 1
            rating_social_list1[i,j] = 1
        elif rating_social_list[i,j] < 15 and rating_social_list[i,j] >= 10:
            nums[2] += 1
            rating_social_list1[i,j] = 2
        elif rating_social_list[i,j] < 20 and rating_social_list[i,j] >= 15:
            nums[3] += 1
            rating_social_list1[i,j] = 2
        elif rating_social_list[i,j] < 25 and rating_social_list[i,j] >= 20:
            nums[4] += 1
            rating_social_list1[i,j] = 2
        elif rating_social_list[i,j] < 30 and rating_social_list[i,j] >= 25:
            nums[5] += 1
            rating_social_list1[i,j] = 3
        elif rating_social_list[i,j] < 35 and rating_social_list[i,j] >= 30:
            nums[6] += 1 
            rating_social_list1[i,j] = 3
        elif rating_social_list[i,j] < 40 and rating_social_list[i,j] >= 35:
            nums[7] += 1
            rating_social_list1[i,j] = 3
        elif rating_social_list[i,j] >= 40:
            nums[8] += 1
            rating_social_list1[i,j] = 4
        else :
            nums[9] += 1
       
       
       
 # [94011, 53517, 32271, 21976, 15702, 11919, 9705, 7968, 40334, 68324]
    # 1 1 2 2 2 3 3 3 4 
'''
for i in range(rows):
    for j in range(cols):
        if rating_social_list[i,j] < 5 and rating_social_list[i,j] > 0:
            nums[0] += 1
            rating_social_list1[i,j] = 1
        elif rating_social_list[i,j] < 10 and rating_social_list[i,j] >= 5:
            nums[1] += 1
            rating_social_list1[i,j] = 2
        elif rating_social_list[i,j] < 15 and rating_social_list[i,j] >= 10:
            nums[2] += 1
            rating_social_list1[i,j] = 3
        elif rating_social_list[i,j] < 20 and rating_social_list[i,j] >= 15:
            nums[3] += 1
            rating_social_list1[i,j] = 4
        elif rating_social_list[i,j] < 25 and rating_social_list[i,j] >= 20:
            nums[4] += 1
            rating_social_list1[i,j] = 4
        elif rating_social_list[i,j] < 30 and rating_social_list[i,j] >= 25:
            nums[5] += 1
            rating_social_list1[i,j] = 4
        elif rating_social_list[i,j] < 35 and rating_social_list[i,j] >= 30:
            nums[6] += 1 
            rating_social_list1[i,j] = 4
        elif rating_social_list[i,j] < 40 and rating_social_list[i,j] >= 35:
            nums[7] += 1
            rating_social_list1[i,j] = 4
        elif rating_social_list[i,j] >= 40:
            nums[8] += 1
            rating_social_list1[i,j] = 4
        else :
            nums[9] += 1
'''
# for i in range(rows):
#     for j in range(cols):
#         if rating_social_list[i,j] < 5:
#             nums[0] += 1
#         elif rating_social_list[i,j] < 10:
#             nums[1] += 1
#         elif rating_social_list[i,j] < 15:
#             nums[2] += 1
#         elif rating_social_list[i,j] < 20:
#             nums[3] += 1
#         elif rating_social_list[i,j] < 25:
#             nums[4] += 1
#         elif rating_social_list[i,j] < 30:
#             nums[5] += 1
#         elif rating_social_list[i,j] < 35:
#             nums[6] += 1
#         elif rating_social_list[i,j] < 40:
#             nums[7] += 1
#         else :
#             nums[8] += 1

# for i in range(rows):
#     for j in range(cols):
#         if rating_social_list[i,j] < 2 and rating_social_list[i,j] > 0:
#             nums[0] += 1
#             rating_social_list1[i,j] = 1
#         elif rating_social_list[i,j] < 3 and rating_social_list[i,j] > 1:
#             nums[1] += 1
#             rating_social_list1[i,j] = 2
#         elif rating_social_list[i,j] < 4 and rating_social_list[i,j] > 2:
#             nums[2] += 1
#             rating_social_list1[i,j] = 2
#         elif rating_social_list[i,j] < 5 and rating_social_list[i,j] > 3:
#             nums[3] += 1
#             rating_social_list1[i,j] = 3
#         elif rating_social_list[i,j] < 6 and rating_social_list[i,j] > 4:
#             nums[4] += 1
#             rating_social_list1[i,j] = 3
#         elif rating_social_list[i,j] < 10 and rating_social_list[i,j] >= 6:
#             nums[5] += 1
#             # rating_social_list1[i,j] = 4
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 15 and rating_social_list[i,j] >= 10:
#             nums[6] += 1
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 20 and rating_social_list[i,j] >= 15:
#             nums[7] += 1
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 25 and rating_social_list[i,j] >= 20:
#             nums[8] += 1
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 30 and rating_social_list[i,j] >= 25:
#             nums[9] += 1
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 35 and rating_social_list[i,j] >= 30:
#             nums[10] += 1 
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 40 and rating_social_list[i,j] >= 35:
#             nums[11] += 1
#             rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] >= 40:
#             nums[12] += 1
#             rating_social_list1[i,j] = 4
#         else :
#             nums[13] += 1


print(nums)
rating_social_list1 = sp.coo_matrix(rating_social_list1) 
print(ratingClass)
ratingClass1 = np.unique(rating_social_list1.data).size #42
print('ratingClass1',ratingClass1)
# print(rating_social_list1)
with open(workdir + args.dataset + '/rating_social.pkl', 'wb') as f:
    pickle.dump(rating_social_list1, f, pickle.HIGHEST_PROTOCOL)
        
# print(rating_social_list)

'''
for u in tqdm(range(user_count + 1)):
    ratings = trust_df[trust_df['uid'] == u]
    u_ratings = ratings['iid'].tolist()
    u_labels = ratings['label'].tolist()
    hist = trust_df[trust_df['uid'] == u]
    u_users = hist['fid'].unique().tolist()
    if len(u_users) == 0 or len(u_ratings) == 0:
        count2 += 1
        continue
    else:
        for i,r in zip(u_ratings,u_labels):
            u = int(u)
            i = int(i)
            r = int(r)
            real_list.append((u,i,r))
        for f in u_users:
            real_trust_list.append((int(u),int(f)))
        # count3 += 1
        # for f in u_users:
            # social_adj_lists[u].add(f)
print('cold_social',count2)



# user1_D = np.array(social_list.sum(axis=1)).squeeze()
# #        print(self.users_D.shape)
# user1_D[user1_D == 0.] = 1
# item1_D = np.array(social_list.sum(axis=0)).squeeze()
# item1_D[item1_D == 0.] = 1.
# print(social_list)

'''
'''


with open(workdir + args.dataset + '/dataset3.pkl', 'wb') as f:
    pickle.dump(train_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_r, f, pickle.HIGHEST_PROTOCOL)

    pickle.dump(test_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_r, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(user_count, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_count, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(adj_mat, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(social_mat, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(adj_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(social_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(multi_adj, f, pickle.HIGHEST_PROTOCOL)
'''    
  