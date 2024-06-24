# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:46:42 2023

@author: lilujun
"""

import json
import numpy as np
import torch.nn.functional as F
import tensorflow
# data = {'name': 'Tom', 'age': 18}
# with open('data.json', 'w') as f:
#     data_str = json.dumps(data)
#     f.write(data_str)
#{"-1": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

# array = np.zeros(100,dtype=int)
# array=[[None] * 100]*100
# print(array)
# # print(array)
# # dicts = {"-%d:, %d:"%(0.01*i,0.01*i) for i in range(1,101)}
# # print(dicts)
rows = 151
cols = 151
dicts = {}
array = [[0 for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        if i == j:
            array[i][j] = 1
            
#print(array)
#     # A=array
for i in range(rows):
    dicts['%.2f'%(0.01*(i))]=[0, 1]+array[i]
    dicts['-%.2f'%(0.01*(i))]=[1, 0]+array[i]
#dicts['%.2f'%(0.00)]=
print(dicts)
with open('charges_default.json', 'w') as f:
    data_str = json.dumps(dicts)
    f.write(data_str)

# # for i in range(100): 
# #     code = F.one_hot(i, num_classes=tensor(100))
# #     print(code)