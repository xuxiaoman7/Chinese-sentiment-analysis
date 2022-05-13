"""
    对cvs格式的文件进行处理
    返回分词好的结果
"""
import jieba
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import warnings
warnings.filterwarnings('ignore')


#分割数据集，70%做训练集，30%做测试集
train_rate = 0.7
#y值(训练集）
train_y = np.concatenate((np.ones(1000),np.zeros(1000)))
np.save('train_y.npy',train_y)

#x值（训练集）
pd_x = pd.read_csv('take-out.csv',encoding='gbk',usecols=['review'])
#分词处理
list_x = []
for i in range(0,2000):
    str = pd_x.loc[i][0]
    words = jieba.lcut(str)
    list_x.append(words)
#训练word2vec模型(求词向量）
model = Word2Vec(list_x,sg=1,vector_size=200,window=5,min_count=2,negative=3,sample=0.001,hs=1,workers=4)
model.save("word2vec_model.pkl")

#求句向量
def sum_vec(text):
    vec = np.zeros(200).reshape((1,200))
    for word in text:
        try:
            vec += model.wv[word].reshape((1,200))
        except KeyError:
            continue
    return vec

train_vec = np.concatenate([sum_vec(z) for z in list_x])
print(train_vec.shape)
train_vec = train_vec.reshape((1400,200))
np.save('train_x_vec.npy',train_vec)
print(train_vec.shape)
