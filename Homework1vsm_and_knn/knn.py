import math
import random
from collections import Counter


# 余弦相似度存放类
class Cosine:
    def __init__(self, index, cos):
        self.index = index
        self.cos = cos


# 加载文档向量
def loaddata(filepath):
    vectors = []
    label = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            vector = []
            item = str(line).split(',')
            for i in range(len(item) - 1):
                vector.append(float(item[i]))
            #print(str(item[len(item) - 2]))
            label.append(str(item[len(item) - 1]).replace('\n', ''))
            vectors.append(vector)
            line = f.readline()
    return vectors, label


# 随机选出测试集 20%
def dataset(label):
    test = []
    train = []
    num = Counter(label)
    begin = 0
    for k, v in num.items():
        end = begin + v
        testi = random.sample(list(range(begin, end)), v // 5)
        for i in testi:
            test.append(i)
        begin = end
    for i in range(len(label)):
        if i not in test:
            train.append(i)
    return train, test


# knn分类器（利用余弦相似度划分）
def classification(train, test, vectors, label, k):
    results = []
    correct = 0
    leng = len(vectors[0])
    for i in test:
        result = []
        result.append(i)
        result.append(label[i])
        topk = []
        for j in range(k):
            topk.append(Cosine(-1, -1))
        minj = 0
        sum=0
        for x in vectors[i]:
            sum+=x
        if sum==0 :
            continue
        for j in train:
            dot = 0
            modi = modj = 0
            for m in range(leng):
                dot = dot + vectors[i][m]*vectors[j][m]
                modi = modi + pow(vectors[i][m], 2)
                modj = modj + pow(vectors[j][m], 2)
            if modj==0 :
                continue
            cosine = dot / (math.sqrt(modi) * math.sqrt(modj))
            cur = Cosine(j, cosine)
            flag = False
            for m in range(k):
                if cur.cos > topk[m].cos:
                    topk[minj] = cur
                    flag = True
                    break
            if flag:
                minj = 0
                for m in range(k):
                    if topk[m].cos < topk[minj].cos:
                        minj = m
        numc = {}
        for j in range(k):
            if label[topk[j].index] in numc:
                numc[label[topk[j].index]] = numc[label[topk[j].index]] + 1
            else:
                numc[label[topk[j].index]] = 1
        ma = sorted(numc.items(), key=lambda x: x[1], reverse=True)
        print(ma)
        result.append(ma[0][0])
        result.append(str(result[1]) == str(ma[0][0]))
        if str(result[1]) == str(ma[0][0]):
            correct = correct + 1
        results.append(result)
        print(str(result[1]) == str(ma[0][0]))
    return results, correct/len(results)


def main():
    filepath = 'vector7.csv'
    vectors, label = loaddata(filepath)
    train, test = dataset(label)
    print(train, '\n', test)
    #k = math.ceil(len(label)/1000)
    k=200
    knn, ap = classification(train, test, vectors, label, k)
    print(knn)
    print(ap)


if __name__ == "__main__":
    main()