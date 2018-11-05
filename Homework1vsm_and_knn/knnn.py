import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn import neighbors
def load_train_data(path):
    train=np.loadtxt(path,delimiter=",", skiprows=0)
    #print(train.shape)
    divide=int((train.shape[0])*0.6)
    #print(divide)
    vec_train=train[0:divide,0:-1]
    vec_test = train[divide:train.shape[0], 0:-1]
    label=train[0:divide,-1]
    labels=[int(i) for i in label]
    print(type(labels)	)
    print(vec_test)
    return vec_test,vec_train,labels
def main():
    print('knn')
    vec_test,vec ,labels = load_train_data("vector7.csv")
    print('################')
    knn = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')  # 取得knn分类器
    print(vec)
    print(vec_test)
    labels = np.array(labels)  #
    print(labels)
    data_test=np.array(vec_test)
    print(data_test)
    knn.fit(vec, labels)  # 导入数据进行训练
    print(knn.predict(data_test))
    #data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    #print(data)
    #labels = np.array([1, 1, 1, 2, 2, 2])  # labels则是对应Romance和Action
    #print(labels)
    #knn.fit(data, labels)  # 导入数据进行训练
    #print(knn.predict([[18, 90]]))
    #print([[18, 90]])


if __name__ == '__main__':
    main()