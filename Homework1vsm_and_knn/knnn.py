import numpy as np
from sklearn import neighbors
key=1
def load_data(path):
    train=np.loadtxt(path,delimiter=",", skiprows=0)
    vec_train=train[0:,0:-1]
    label=train[0:,-1]
    labels=[int(i) for i in label]
    #print(type(labels)	)
    return vec_train,labels
def main():
    print('knn')
    vec_train,labels1 = load_data("vector.csv")
    vec_test,labels2 = load_data("vector7.csv")
    knn = neighbors.KNeighborsClassifier(n_neighbors=key, metric='euclidean')  # 取得knn分类器
    print('train data')
    print(vec_train)
    print('test data')
    print(vec_test)
    labels1 = np.array(labels1)  #
    labels2 = np.array(labels2)
    print("the true value")
    print(labels2)
    knn.fit(vec_train, labels1)  # 导入数据进行训练
    predict = knn.predict(vec_train)
    result  = predict[0:labels2.shape[0]]
    print("the predict value")
    print(result)
    acc=0
    for i in range(labels2.shape[0]):
        if labels2[i]==result[i]:
            acc+=1
    accuracy=acc/labels2.shape[0]
    print("the accuracy of the knn with k=",key,' : ',accuracy)
if __name__ == '__main__':
    main()