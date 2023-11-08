# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:43:37 2021

@author: 54398
"""


from torch.utils.data import Dataset, DataLoader
import pickle
import torch

import scipy.sparse as sp
import numpy as np


class BasicData(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getInfo(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
        
    def getSocialSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class BasicDataset(BasicData):
    def __init__(self):
        print("init dataset")
        with open('/data1/lss/project/RGCN/datasets_pre/Epinions' + '/dataset4.pkl', 'rb') as f:
            self.train_u = pickle.load(f)
            self.train_v = pickle.load(f)
            self.train_r = pickle.load(f)
            self.valid_u = pickle.load(f)
            self.valid_v = pickle.load(f)
            self.valid_r = pickle.load(f)
            self.test_u = pickle.load(f)
            self.test_v = pickle.load(f)
            self.test_r = pickle.load(f)
            self.user_count = pickle.load(f)
            self.item_count = pickle.load(f)
            self.adj_mat = pickle.load(f)
            self.social_mat = pickle.load(f)   
            self.train_list = pickle.load(f) 
            self.test_list = pickle.load(f)
            self.adj_list = pickle.load(f)
            self.social_list = pickle.load(f)
            self.multi_social = pickle.load(f)
            self.multi_adj = pickle.load(f)
        
        # print(self.adj_list)
        self.multi_adj_new = self.create_multi_adj_new(self.multi_adj)
        self.multi_social_new = self.create_multi_social_new(self.multi_social)
        print('multi_social',self.multi_social.shape)
        print('multi_social_new',self.multi_social_new.shape)
        print('multi_adj',self.multi_adj.shape)
        print('multi_adj_new',self.multi_adj_new.shape)
        
        # multi_adj = self.create_multi_adj()
        # print(multi_adj)
        
    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()
    
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
        
    def create_multi_adj_new(self,multi_adj):
        # 重新设计multi_adj矩阵，尺寸为n_user+5*n_item * n_user+5*n_item
        a = sp.csr_matrix((multi_adj.shape[1], multi_adj.shape[1]))
        b = sp.csr_matrix((multi_adj.shape[0], multi_adj.shape[0]))
        multi_uv_adj = sp.vstack([sp.hstack([a, multi_adj.T]), sp.hstack([multi_adj,b])])
        multi_adj_new = self.normalize_adj(multi_uv_adj + sp.eye(multi_uv_adj.shape[0])) 
        self.adj_sp_tensor = self.sparse_mx_to_torch_sparse_tensor(multi_adj_new).cuda()
        
        return self.adj_sp_tensor
    def create_multi_social_new(self,multi_social_adj):
        # 重新设计multi_adj矩阵，尺寸为n_user*4 * n_user*4
        a = sp.csr_matrix((multi_social_adj.shape[1], multi_social_adj.shape[1]))
        b = sp.csr_matrix((multi_social_adj.shape[0], multi_social_adj.shape[0]))
        multi_uv_adj = sp.vstack([sp.hstack([a, multi_social_adj.T]), sp.hstack([multi_social_adj,b])])
        multi_adj_new = self.normalize_adj(multi_uv_adj + sp.eye(multi_uv_adj.shape[0])) 
        self.adj_sp_tensor = self.sparse_mx_to_torch_sparse_tensor(multi_adj_new).cuda()
        
        
        return self.adj_sp_tensor
    
    def create_multi_adj(self):
        trainMat = self.adj_list
        
        ratingClass = np.unique(trainMat.data).size
        userNum, itemNum = trainMat.shape
        mult_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
        uidList = trainMat.tocoo().row
        iidList = trainMat.tocoo().col
        rList = trainMat.tocoo().data

        for i in range(uidList.size):
            uid = uidList[i]
            iid = iidList[i]
            r = rList[i]
            mult_adj[iid*ratingClass+r-1, uid] = 1
            
        mult_adj_s = mult_adj.tocsr()

        return mult_adj_s
    def create_multi_social(self):
        trainMat = self.multi_social
        
        ratingClass = np.unique(trainMat.data).size
        userNum, itemNum = trainMat.shape
        mult_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
        uidList = trainMat.tocoo().row
        iidList = trainMat.tocoo().col
        rList = trainMat.tocoo().data

        for i in range(uidList.size):
            uid = uidList[i]
            iid = iidList[i]
            r = rList[i]
            mult_adj[iid*ratingClass+r-1, uid] = 1
            
        mult_adj_s = mult_adj.tocsr()

        return mult_adj_s
    
    def n_users(self):
        return self.user_count
        
    
    def getInfo(self):
        return self.train_u,self.train_v,self.train_r,self.valid_u,self.valid_v,self.valid_r,self.test_u,self.test_v,self.test_r,self.user_count,self.item_count,self.multi_social_new,self.multi_adj_new
    
    
    def m_items(self):
        return self.item_count
    

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        # A = sp.dok_matrix((self.user_count ,
        #                self.item_count ), dtype=np.float32)
        # A = A.tolil()
        # adj_matrix = self.adj_list.tolil()
        # A[:self.user_count, :self.item_count] = adj_matrix
    
        # # A = A.todok()
        # # # norm adj matrix
        # # sumArr = (A > 0).sum(axis=1)
        # # # add epsilon to avoid Devide by zero Warning
        # # diag = np.array(sumArr.flatten())[0] + 1e-7
        # # diag = np.power(diag, -0.5)
        # # D = sp.diags(diag)
        # L =  A 
        # # covert norm_adj matrix to tensor
        # L = sp.coo_matrix(L)
        
        return self.adj_list
        
        
        

    def getSocialSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        
        # A = sp.dok_matrix((self.user_count ,
        #                self.user_count ), dtype=np.float32)
        # A = A.tolil()
        # social_matrix = self.social_list.tolil()
        # A[:self.user_count, :self.user_count] = social_matrix
    
        # # A = A.todok()
        # # # norm adj matrix
        # # sumArr = (A > 0).sum(axis=1)
        # # # add epsilon to avoid Devide by zero Warning
        # # diag = np.array(sumArr.flatten())[0] + 1e-7
        # # diag = np.power(diag, -0.5)
        # # D = sp.diags(diag)
        # L =  A 
        # # covert norm_adj matrix to tensor
        # L = sp.coo_matrix(L)
        return self.social_list

    
    def getTensorSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        return self.adj_mat.to('cuda')
    
    def getTensorSocialSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        return self.social_mat.to('cuda')
dataset = BasicDataset()
