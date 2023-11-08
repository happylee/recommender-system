# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:36:19 2021

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


##

#dir_data = 'G:/recommender/pytorch/GraphRec_test/origin_code/datasets_pre/'

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
 	click_f = np.loadtxt(workdir+'Epinions/rating.txt', dtype = np.int32)
 	trust_f = np.loadtxt(workdir+'Epinions/trust.txt', dtype = np.int32)
# 	click_f = loadmat(workdir + 'Epinions/rating.mat')['rating']
# 	trust_f = loadmat(workdir + 'Epinions/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
    click_f = np.loadtxt(workdir+'Epinions/rating.txt', dtype = np.int32)
    trust_f = np.loadtxt(workdir+'Epinions/social.txt', dtype = np.int32)
    train_f = np.loadtxt(workdir+'Epinions/train.txt', dtype = np.int32)
    valid_f = np.loadtxt(workdir+'Epinions/valid.txt', dtype = np.int32)
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
valid_list = []
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
 
for s in valid_f:
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
    valid_list.append([uid, iid, label])
 
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



            
            
# print(train_f)

user_count = n_user + 1
item_count = m_item + 1
print(user_count)

train_u = []
train_v = []
train_r = []
valid_u = []
valid_v = []
valid_r = []
test_u = []
test_v = []
test_r = []

trust_uid = []
trust_fid = []


for i,data in enumerate(train_f):
    train_u.append(data[0])
    train_v.append(data[1])
    train_r.append(data[2])
    
for i,data in enumerate(valid_f):
    valid_u.append(data[0])
    valid_v.append(data[1])
    valid_r.append(data[2])

for i,data in enumerate(test_f):
    test_u.append(data[0])
    test_v.append(data[1])
    test_r.append(data[2])
    
for i,data in enumerate(trust_f):
    trust_uid.append(data[0])
    trust_fid.append(data[1])
    
adj_list = sp.coo_matrix((train_r,(train_u,train_v)),shape=(user_count,item_count),dtype=np.float32)



users_D = np.array(adj_list.sum(axis=1)).squeeze()
#        print(self.users_D.shape)
# users_D[users_D == 0.] = 1
# items_D = np.array(adj_list.sum(axis=0)).squeeze()
# items_D[items_D == 0.] = 1.
# print(adj_list)


social_list = sp.coo_matrix((np.ones(len(trust_uid)),(trust_uid,trust_fid)),shape=(user_count,user_count),dtype=np.float32)
print('social',social_list)

# user1_D = np.array(social_list.sum(axis=1)).squeeze()
# #        print(self.users_D.shape)
# user1_D[user1_D == 0.] = 1
# item1_D = np.array(social_list.sum(axis=0)).squeeze()
# item1_D[item1_D == 0.] = 1.
# print(social_list)



def get_norm_social_adj_mat(social_list):
    #build adj_matrix
    A = sp.dok_matrix((user_count ,
                       user_count), dtype=np.float32)
    A = A.tolil()
    social_matrix = social_list.tolil()
    A[:user_count, :user_count] = social_matrix
    
    # A = A.todok()
    # # norm adj matrix
    # sumArr = (A > 0).sum(axis=1)
    # # add epsilon to avoid Devide by zero Warning
    # diag = np.array(sumArr.flatten())[0] + 1e-7
    # diag = np.power(diag, -0.5)
    # D = sp.diags(diag)
    L =  A 
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    # print(SparseL.todense())
    return SparseL

def get_norm_adj_mat(adj_list):
    #build adj_matrix
    A = sp.dok_matrix((user_count + item_count,
                       user_count + item_count), dtype=np.float32)
    A = A.tolil()
    interaction_matrix = adj_list.tolil()
    A[:user_count, user_count:] = interaction_matrix
    A[user_count:, :user_count] = interaction_matrix.transpose()
    A = A.todok()
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid Devide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    # print(SparseL.todense())
    return SparseL

adj_mat = get_norm_adj_mat(adj_list)
# print(adj_mat)
social_mat = get_norm_social_adj_mat(social_list)
# print(social_mat)


def create_multi_adj(adj_list):
    trainMat = adj_list
    
    ratingClass = 5#np.unique(trainMat.data).size
    print('ratingClass',ratingClass)
    print(np.unique(trainMat.data))
    userNum, itemNum = trainMat.shape
    mult_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
    uidList = trainMat.tocoo().row
    iidList = trainMat.tocoo().col
    rList = trainMat.tocoo().data
    print(mult_adj.shape)

    for i in range(uidList.size):
        uid = uidList[i]
        iid = iidList[i]
        r = rList[i]
        if (iid * ratingClass + r - 1) > 5 * itemNum:
            continue
        else:
            mult_adj[iid * ratingClass + r - 1, uid] = 1
        # mult_adj[iid*ratingClass+r-1, uid] = 1 #row index (1481395) out of rang
        
    mult_adj_s = mult_adj.tocsr()

    return mult_adj_s


def create_multi_social_adj(adj_list):
    trainMat = adj_list
    
    ratingClass = np.unique(trainMat.data).size
    print('ratingClass',ratingClass)
    userNum, itemNum = trainMat.shape
    mult_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
    uidList = trainMat.tocoo().row
    iidList = trainMat.tocoo().col
    rList = trainMat.tocoo().data

    for i in range(uidList.size):
        uid = uidList[i]
        iid = iidList[i]
        r = rList[i]
        if (iid*ratingClass+r-1) > 5*itemNum:
            continue
        else:
            mult_adj[iid*ratingClass+r-1, uid] = 1
        
    mult_adj_s = mult_adj.tocsr()

    return mult_adj_s


multi_adj = create_multi_adj(adj_list)#(198475, 2964)
print(multi_adj.shape)


with open('/data1/lss/project/RGCN/datasets_pre/Epinions' + '/rating_social.pkl', 'rb') as f:
    multi_social_list = pickle.load(f)

multi_social_adj = create_multi_social_adj(multi_social_list)#douban(198475, 2964)
print(multi_social_adj.shape)

with open(workdir + args.dataset + '/dataset4.pkl', 'wb') as f:
    pickle.dump(train_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_r, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_r, f, pickle.HIGHEST_PROTOCOL)

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
    pickle.dump(multi_social_adj, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(multi_adj, f, pickle.HIGHEST_PROTOCOL)
    
     

# print(adj_list)
# =============================================================================
# #分割数据集，并将训练集，测试集存储到pickle文件中
# =============================================================================
# for s in click_f:
#     uid = s[0]
#     iid = s[1]
#     if args.dataset == 'toy':
#         label = s[2]
#     else:
#         label = s[2]
        
#     if uid > user_count :
#         user_count = uid
#     if iid > item_count :
#         item_count = iid
#     if label > rate_count :
#         rate_count = label
#     click_list.append([uid, iid, label])

    
'''

trust_list = []  
for s in trust_f:
    uid = s[0]
    fid = s[1]
    # if args.dataset == 'toy':
        # label = s[2]
    # else:
        # label = s[2]
        
    # if uid > user_count :
        # user_count = uid
    # if iid > item_count :
        # item_count = iid
    # if label > rate_count :
        # rate_count = label
    trust_list.append([uid, fid])


# social_adj = sp.coo_matrix((np.ones()))
# user_count = int(user_count)
# item_count = int(item_count)
user_count += 1
item_count += 1
print(user_count)
print(item_count)
print(rate_count)
pos_list = []



for i in range(len(click_list)):
 	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
    
 
pos_list = list(set(pos_list))
print('pos_list',len(pos_list))
random.shuffle(pos_list)
    
print('interaction',len(pos_list))
# train, valid and test data split
# random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2*num_test]
train_set = pos_list[2*num_test:]
# traindata_set = pos_list[num_test:]

# traindata_set = pos_list[num_test:]
train_u = []
train_v = []
train_r = []

valid_u = []
valid_v = []
valid_r = []

test_u = []
test_v = []
test_r = []

social_u = []
social_f = []
# train_user, train_item, train_label = []
for i,data in enumerate(train_set):
    train_u.append(data[0])
    train_v.append(data[1])
    train_r.append(data[2])
    
for i,data in enumerate(trust_list):
    social_u.append(data[0])
    social_f.append(data[1])
    

adj_list = sp.coo_matrix((train_r, (train_u, train_v)),shape=(user_count, item_count))
# print(adj)
social_adj = sp.coo_matrix((np.ones(len(social_u)),(social_u,social_f)),shape=(user_count,user_count))


# adj = torch.FloatTensor(np.array(adj.todense()))



for i,data in enumerate(valid_set):
    valid_u.append(data[0])
    valid_v.append(data[1])
    valid_r.append(data[2])

for i,data in enumerate(test_set):
    test_u.append(data[0])
    test_v.append(data[1])
    test_r.append(data[2])
    
with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
    pickle.dump(train_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_r, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_r, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_u, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_v, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_r, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(user_count, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_count, f, pickle.HIGHEST_PROTOCOL)
    
train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])

test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
'''
 # train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
'''

index_dict = defaultdict(list)

 # =============================================================================
 # history_u_lists, 
 # history_ur_lists, 
 # history_v_lists, 
 # history_vr_lists, 
 # train_u, train_v, train_r, 
 # test_u, test_v, test_r, 
 # social_adj_lists, ratings_list
 # =============================================================================
history_u_lists = defaultdict(list)
history_ur_lists = defaultdict(list)
for u in tqdm(range(user_count + 1)):
    hist = click_df[click_df['uid'] == u]
    u_items = hist['iid'].tolist()
    u_ratings = hist['label'].tolist()
    if u_items == []:
         # i = 0
        continue
    else:
        history_u_lists[u] = u_items
        history_ur_lists[u] = u_ratings
        
         # u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])
        
 # print(history_u_lists)

history_v_lists = defaultdict(list)
history_vr_lists = defaultdict(list)

for v in tqdm(range(item_count + 1)):
    hist = click_df[click_df['iid'] == v]
    v_users = hist['uid'].tolist()
    v_ratings = hist['label'].tolist()
    if v_users == []:
         # i = 0
        continue
    else:
        history_v_lists[v] = v_users
        history_vr_lists[v] = v_ratings
        
social_adj_lists =  defaultdict(set)
ratings_list =defaultdict(set)
trust_list = []
for s in trust_f:
  uid = s[0]
  fid = s[1]
  if uid > user_count or fid > user_count:
         continue
  trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
# trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')
 
for u in tqdm(range(user_count + 1)):
    hist = trust_df[trust_df['uid'] == u]
    u_users = hist['fid'].unique().tolist()
    if u_users == []:
        continue
    else:
        for f in u_users:
            social_adj_lists[u].add(f)
 # 		u_users_list.append(u_users)
 # 		uu_items = []
 # 		for uid in u_users:
 # 			uu_items.append(u_items_list[uid])
 # 		u_users_items_list.append(uu_items)
 # print(social_adj_lists)
ratings_list = {2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}


with open(workdir + args.dataset + '/history_lists.pkl', 'wb') as f:
    pickle.dump(history_u_lists, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(history_ur_lists, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(history_v_lists, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(history_vr_lists, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(social_adj_lists, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(ratings_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(adj_list, f, pickle.HIGHEST_PROTOCOL)
'''
