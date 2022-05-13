import jieba
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import warnings
warnings.filterwarnings('ignore')  #ignore warning
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib .pyplot as plt
from sklearn.tree import plot_tree

#CART模型
#导入词向量作为训练特征
x = np.load('train_x_vec.npy')
#导入情绪分类作为目标特征
y = np.load('train_y.npy')

#设置最大深度，深度过深容易过拟合
model = DecisionTreeClassifier(max_depth=20)

#采用10折交叉验证法训练模型
kf = KFold(n_splits=10,shuffle=True,random_state=0)
curr_acc = 0
for train_index,test_index in kf.split(x):
    clt = model.fit(x[train_index],y[train_index])
    y_predict = model.predict(x[test_index])
    curr_acc = curr_acc + accuracy_score(y_predict,y[test_index])
print("模型准确率为:",curr_acc/10)

#数据可视化
plt.figure()
plot_tree(model,filled=True)
plt.show()

#保存模型为二进制文件
joblib.dump(model,'dt_model.pkl')
