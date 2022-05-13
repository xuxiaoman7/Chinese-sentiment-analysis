from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#加载数据
#导入词向量作为训练特征
x = np.load('train_x_vec.npy')
#导入情绪分类作为目标特征
y = np.load('train_y.npy')

model = RandomForestClassifier()

#因为随机森林的算法特性，本来无需交叉验证；但是实验要求为10折交叉验证，且已有理论证明二者所得准确率一致（数据集平衡的情况下），故使用交叉验证
kf = KFold(n_splits=10,shuffle=True,random_state=0)
curr_acc = 0
for train_index,test_index in kf.split(x):
    clt = model.fit(x[train_index],y[train_index])
    y_predict = model.predict(x[test_index])
    curr_acc = curr_acc + accuracy_score(y_predict,y[test_index])

print("模型准确率为：",curr_acc/10)

np.save('rf_model.pkl',model)