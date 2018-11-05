import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn import neighbors
def load_data(path):
    train=np.loadtxt(path,delimiter=",", skiprows=0)
    vec_train=train[0:,0:5]
    label=train[0:,-1]
    labels=[int(i) for i in label]
    #print(type(labels)	)
    return vec_train,labels
def main():
    print('knn')
    vec_train,labels1 = load_data("vector7.csv")
    vec_test,labels2 = load_data("vector8.csv")
    #vec_test,labels2=
    knn = neighbors.KNeighborsClassifier(n_neighbors=2, metric='euclidean')  # 取得knn分类器
    print(vec_train)
    print(vec_test)
    labels1 = np.array(labels1)  #
    labels2 = np.array(labels2)
    print(labels1)
    print(labels2)
    knn.fit(vec_train, labels1)  # 导入数据进行训练
    #print(knn.predict(vec_train))
if __name__ == '__main__':
    main()