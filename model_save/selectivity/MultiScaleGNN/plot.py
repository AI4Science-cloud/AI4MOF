# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:18:39 2024

@author: lilujun
"""

# import pandas as pd  
# import matplotlib.pyplot as plt  
# import numpy as np  
  
# # 读取CSV文件  
# df = pd.read_csv('my_train_job_test_outputs.csv')  # 替换为你的CSV文件名  
  
# # 假设CSV文件的列名分别为'X', 'Y', 和 'Color'  
# x = df['target']  
# y = df['prediction']  
# color_values = df['target']  
  
# # 创建一个颜色映射，将颜色值映射到具体的颜色上  
# # 这里我们假设Color列的值范围在0到1之间，并使用'viridis'颜色映射  
# # 如果Color列的值不在这个范围内或者你想使用不同的颜色映射，请相应地调整  
# norm = plt.Normalize(color_values.min(), color_values.max())  
# cmap = plt.get_cmap('viridis')  
# colors = cmap(norm(color_values))  
  
# # 绘制散点图  
# plt.scatter(x, y, c=colors)  

# # 添加y=x的黑色虚线  
# plt.plot([df['target'].min(), df['target'].max()], [df['target'].min(), df['target'].max()], 'k--')  
  
  
# # 添加颜色条以显示颜色如何映射到Color列的值  
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  
# sm.set_array([])  
# plt.colorbar(sm, label='Color')  
  
# # 设置坐标轴标签和图表标题  
# plt.xlabel('target')  
# plt.ylabel('prediction')  
# plt.title('Scatter Plot with Color Mapped to Third Column')  
  
# # 显示图表  
# plt.show()
import pandas as pd  
import matplotlib.pyplot as plt 
import matplotlib 

# 设置全局字体属性  
matplotlib.rcParams['font.family'] = 'Times New Roman'  # 字体类型  
matplotlib.rcParams['font.size'] = 14         # 字体大小
  
# 假设你的CSV文件在一个文件夹中，文件名为"data1.csv", "data2.csv", ...  
csv_files = ["MultiFormer_my_train_job_train_outputs.csv", "MultiFormer_my_train_job_val_outputs.csv", "MultiFormer_my_train_job_test_outputs.csv"]  # 这里替换为你的CSV文件名  
colors = ['red', 'green', 'blue']  # 为每个CSV文件设置不同的颜色  
labels = ['train set', 'val set', 'test set']  # 为每个CSV文件设置不同的标签 
maxrange = 450  
# 读取CSV文件并绘制散点图  
for i, csv_file in enumerate(csv_files):  
    # 读取CSV文件  
    df = pd.read_csv(csv_file)  
      
    # 假设我们使用"column_x"作为X轴，"column_y"作为Y轴，你需要替换为实际的列名  
    x = df['target']  
    y = df['prediction']  
      
    # 绘制散点图，并使用不同的颜色  
    #plt.scatter(x, y, label=f'Data from {csv_file}', color=colors[i]) 
    plt.scatter(x, y, label=labels[i], color=colors[i])   

plt.xlim(0, maxrange)  
plt.ylim(0, maxrange)

# 添加y=x的黑色虚线  
x_line = [0, maxrange]  
y_line = [0, maxrange]
#plt.plot([df['target'].min(), df['target'].max()], [df['target'].min(), df['target'].max()], 'k--')  
plt.plot(x_line, y_line , 'k--')  
 
# 设置图表的标题和轴标签  
#plt.title('Scatter Plot from Multiple CSV Files',fontname = 'Times New Roman')  
plt.xlabel('target',fontname = 'Times New Roman',fontsize=14) 
plt.ylabel('prediction', fontname = 'Times New Roman',fontsize=14)  

# 创建自定义的图例句柄和标签  
handles, labels = plt.gca().get_legend_handles_labels()  
custom_labels = [f'{lbl}' for lbl in labels]  
  


# 添加黑色加粗字体的小标签  
plt.text(0.165, 0.97, 'MultiScaleGNN', transform=plt.gca().transAxes,  
         fontsize=14, fontweight='bold', va='top', ha='center')  
plt.text(0.78, 0.25, 'Train       Val       Test', transform=plt.gca().transAxes,  
         fontsize=14, fontname = 'Times New Roman', va='top', ha='center')  
plt.text(0.75, 0.18, '$R^2$ 0.6648 0.6087 0.6391', transform=plt.gca().transAxes,  
         fontsize=14, fontname = 'Times New Roman', va='top', ha='center')  
plt.text(0.72, 0.09, '$RMSE$ 16.7106 18.5491 17.7671', transform=plt.gca().transAxes,  
         fontsize=14, fontname = 'Times New Roman', va='top', ha='center')  
 

# 显示图例，使用自定义的标签  
plt.legend(handles, custom_labels, loc='upper left', bbox_to_anchor=(0.03, 0.9), borderaxespad=0.)

# 显示图表  
plt.show()

# 保存图表为高清PNG图片，设置dpi为300  
plt.savefig('MultiScaleGNN_selectivity.png', dpi=300)
