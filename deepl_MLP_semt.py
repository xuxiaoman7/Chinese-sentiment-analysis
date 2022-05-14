from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#CART模型
#导入词向量作为训练特征
x = np.load('train_x_vec.npy')
#导入情绪分类作为目标特征
y = np.load('train_y.npy')

#定义模型
#输入层的维度为词向量的维度(100，)
def define_model():
    model = Sequential()
    #第一层：50个神经元，relu函数
    model.add(Dense(50,input_shape=(100,),activation='relu'))
    #输出层：一维结果
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model



#10折交叉验证
kf = KFold(n_splits=10,shuffle=True,random_state=0)
curr_acc = 0
for train_index,test_index in kf.split(x):
    clt = model.fit(x[train_index],y[train_index],epochs=10,verbose=2)
    #y_predict = model.predict(x[test_index])
    #curr_acc = curr_acc + accuracy_score(y_predict,y[test_index])
    #print(clt.score(x[test_index],y[test_index]))
    loss,acc = model.evaluate(x[test_index],y[test_index],verbose=0)
    curr_acc = curr_acc+acc
print("模型准确率为:",curr_acc/10)