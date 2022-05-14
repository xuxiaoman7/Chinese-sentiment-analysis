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

#加载数据
#导入词向量作为训练特征
x = np.load('train_x_vec.npy')
#导入情绪分类作为目标特征
y = np.load('train_y.npy')

clf_tree = DecisionTreeClassifier(max_depth=15,random_state=1)

#adaboost实现
def my_adaboost_clf(x_train,y_train,x_test,y_test,M=3,weak_clf=DecisionTreeClassifier(max_depth=15)):
    n_train,n_test = len(x_train),len(x_test)
    #初始化权重
    w = np.ones(n_train)/n_train
    pred_train,pred_test = [np.zeros(n_train),np.zeros(n_test)]

    for i in range(M):
        #使用指定的权重初始分类器
        #检查w是否有nan值
        if(np.any(np.isnan(w))):
            #print(w)
            sum0,count = 0,0
            list = []
            for i in range(len(w)):
                if(np.isnan(w[i])):
                    list.append(i)
                else:
                    count = count + 1
                    sum0 = w[i] + sum0
            if(count == 0):
                mean = 0
            else:
                mean = sum0/count
            for i in list:
                w[i] = mean
            #print(w)
        weak_clf.fit(x_train,y_train,sample_weight=w)
        pred_train_i = weak_clf.predict(x_train)
        pred_test_i = weak_clf.predict(x_test)

        miss = [int(x) for x in (pred_train_i != y_train)]
        #print("weak_clf_%02d reain acc: %.4f"%(i+1,1-sum(miss)/n_train))

        err_m = np.dot(w,miss)
        alpha_m = 0.5*np.log((1-err_m)/float(err_m))

        #新的权重
        miss2 = [x if x==1 else -1 for x in miss]
        w = np.multiply(w,np.exp([float(x)*alpha_m for x in miss2]))
        w = w /sum(w)

        pred_train_i = [1 if x==1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x==1 else -1 for x in pred_test_i]
        pred_train = pred_train+np.multiply(alpha_m,pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m,pred_test_i)

    pred_train = (pred_train>0)*1
    pred_test = (pred_test>0)*1

    #print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == y_test) / n_test))
    return (sum(pred_test == y_test) / n_test)



#采用10折交叉验证法训练模型
kf = KFold(n_splits=10,shuffle=True,random_state=0)
curr_acc = 0
for train_index,test_index in kf.split(x):
    curr_acc = curr_acc + my_adaboost_clf(x[train_index],y[train_index],x[test_index],y[test_index])
print("10折交叉验证后模型准确率为:",curr_acc/10)
