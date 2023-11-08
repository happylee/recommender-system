# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:34:08 2021

@author: 54398
"""
import scipy.sparse as sp
import numpy as np
import pickle

# =============================================================================
# 测试矩阵的创建
# user_count = 2010
#         
# rating_social_list = sp.coo_matrix(np.zeros((user_count,user_count),dtype=np.float32))
# print(rating_social_list.todense())
# # print(rating_social_list.shape)
# =============================================================================

# =============================================================================
# # 测试两个数组中相同元素的个数
# array1 = [1,2,3,4,6]
# array2 = [3,4,5,6,7,8,9]
# print(len(set(array1) & set(array2)))
# =============================================================================

# =============================================================================
# # 测试数组赋值
# nums = [0,0,0,0,0,0,0,0,0]
# 
# nums[0] += 1
# print(nums)
# =============================================================================

# =============================================================================
# # 测试if语句
# nums = [0,0,0,0,0,0,0,0,0]
# 
# rows = 2965
# cols = 2965
# array = np.arange(2965*2965).reshape(2965,2965)
# # array =np.array([[2,55,66,77,88],
# #          [22,1,5,7,9],
# #          [1,5,6,7,8],
# #          [11,22,33,44,55],
# #          [11,22,33,44,55]])
# 
# for i in range(rows):
#     for j in range(cols):
#         if array[i,j] < 5:
#             nums[0] += 1
#         elif array[i,j] < 10 and array[i,j] >= 5:
#             nums[1] += 1
#         elif array[i,j] < 15 and array[i,j] >= 10:
#             nums[2] += 1
#         elif array[i,j] < 20 and array[i,j] >= 15:
#             nums[3] += 1
#         elif array[i,j] < 25 and array[i,j] >= 20:
#             nums[4] += 1
#         elif array[i,j] < 30 and array[i,j] >= 25:
#             nums[5] += 1
#         elif array[i,j] < 35 and array[i,j] >= 30:
#             nums[6] += 1
#         elif array[i,j] < 40 and array[i,j] >= 35:
#             nums[7] += 1
#         else :
#             nums[8] += 1
#             
# print(nums)
# =============================================================================


# 测试矩阵
with open('G:/recommender/pytorch/owncode/nanshou/datasets_pre/douban' + '/rating_social.pkl', 'rb') as f:
    multi_adj = pickle.load(f)
    
print(multi_adj)

# for i in range(rows):
#     for j in range(cols):
#         if rating_social_list[i,j] < 2 and rating_social_list[i,j] > 0:
#             nums[0] += 1
#             # rating_social_list1[i,j] = 1
#         elif rating_social_list[i,j] < 3 and rating_social_list[i,j] > 1:
#             nums[1] += 1
#         elif rating_social_list[i,j] < 4 and rating_social_list[i,j] > 2:
#             nums[2] += 1
#         elif rating_social_list[i,j] < 5 and rating_social_list[i,j] > 3:
#             nums[3] += 1
#         elif rating_social_list[i,j] < 6 and rating_social_list[i,j] > 4:
#             nums[4] += 1
#         elif rating_social_list[i,j] < 10 and rating_social_list[i,j] > 6:
#             nums[5] += 1
#             # rating_social_list1[i,j] = 2
#         elif rating_social_list[i,j] < 15 and rating_social_list[i,j] >= 10:
#             nums[6] += 1
#             # rating_social_list1[i,j] = 3
#         elif rating_social_list[i,j] < 20 and rating_social_list[i,j] >= 15:
#             nums[7] += 1
#             # rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 25 and rating_social_list[i,j] >= 20:
#             nums[8] += 1
#             # rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 30 and rating_social_list[i,j] >= 25:
#             nums[9] += 1
#             # rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 35 and rating_social_list[i,j] >= 30:
#             nums[10] += 1 
#             # rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] < 40 and rating_social_list[i,j] >= 35:
#             nums[11] += 1
#             # rating_social_list1[i,j] = 4
#         elif rating_social_list[i,j] >= 40:
#             nums[12] += 1
#             # rating_social_list1[i,j] = 4
#         else :
#             nums[13] += 1

