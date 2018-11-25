import os
import nltk
import re
import string
import random
from collections import Counter
import numpy as np
rootpath = '../data'
docs = []#存储处理好的文档
words = []#词汇表
dict = [] #词典表(过滤后）
labels = []#标签
countlist = {}#计算单词在所有文档出现的次数
threshold_value_low = 30
threshold_value_high = 800
trainset = []
testset = []
trainlabel = []
testlabel = []
def dataprocessing(rootpath):
    print('input data')
    label = 0
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            slabel=str(label)
            print(slabel+'current file:' + file)
            doc = []  # 存储处理好的文档
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):  # 是文件的话读取文件内容
                with open(filepath, mode='r', encoding='latin-1', errors="ignore") as f:
                    document = f.read()
                f.close()
                # 文件内容处理
                document = document.lower()  # 大写转小写
                tokens = nltk.word_tokenize(document)  # tokenizaton
                punctuation_remove = re.compile('[%s]' % re.escape(string.punctuation))  # 去标点符号
                tokens = list(filter(lambda word: word != "", [punctuation_remove.sub("", word) for word in tokens]))
                # print(tokens)
                stopwords = set(nltk.corpus.stopwords.words("english"))  # 停用词
                stem = nltk.stem.LancasterStemmer()  # 提取词干
                # print(tokens)
                for token in tokens:
                    if not token.isdigit() and token.isalpha():  # 去除数字
                        token = stem.stem(token)  # Stemming
                        if token not in stopwords:  # 去停用词
                            doc.append(token)  # 存储处理好的文章
                            if token not in words:  # 去重
                                words.append(token)
                            if token not in countlist:
                                countlist[token] = 1
                            else:
                                countlist[token] += 1
                labels.append(label)  # 为文章打标签
                docs.append(doc)
        label += 1  # 一个文件夹为一个标签
    #print(docs)
    #print(words)
    print(labels)
    for word in words:  # 过滤的word
        cf = countlist[word]
        #print(word, 'cf=', cf)
        if cf > threshold_value_low and cf < threshold_value_high:
            dict.append(word)
    with open('dict.txt', 'w', errors="ignore") as f:
        for w in dict:
            f.write(w + '\n')
    f.close()
    print('output wordlist finish')
# 单词文档频率
def df(word):
    count = 0
    for i in docs:
        if word in set(i):
            count = count + 1
    return count


# 每个类选取80%作为训练集,20%作为测试集
def filedivide(label):
    test = []
    train = []
    num = Counter(label)
    print(num)
    print(num.items())
    begin = 0
    # 统计每一类的文档数
    for k, v in num.items():
        end = begin + v
        # 从每一类里随机选取20的测试集
        testi = random.sample(list(range(begin, end)), v // 5)
        for i in testi:
            test.append(i)

        begin = end
    # 将剩下的加入到训练集
    for i in range(len(label)):
        if i not in test:
            train.append(i)
    return train, test


# 统计词频，并返回每一篇文档的所有单词的词频向量
def tf(train):
    vector = [0]*len(words)
    for i in train:
        if i in words:
            vector[words.index(i)] += 1
    return vector


# 利用训练集进行训练
def trainbayes(trainset, trainlabel):
    # st 训练集文档数
    st = len(trainset)
    # sw 每一篇文档的维度，即单词表总数
    sw = len(trainset[0])
    # num 类别总数，这里一共有20类，标号0-19
    num = Counter(trainlabel)
    # priori 每个类文档的先验概率，这里采用numpy存储，为了加快计算速度
    priori = np.zeros(len(num), dtype=float)
    for k, v in num.items():
        # 将每个类的文档数赋值给先验概率对应的位置
        priori[int(k)] = v
    # 将每个类的文档数除以文档总数作为每个类出现的先验概率
    priori = priori/len(trainlabel)
    # pn 每个类中，每个单词的词频，最后加1，是个二维数组
    pn = np.ones((len(num), sw), dtype=float)
    # ps 每个类的单词总数与单词表长度之和，这样做是为了防止出现0，是的值为0，即进行平滑处理
    ps = np.ones(len(num), dtype=float)*sw
    # 遍历所有文档，并计算pn和ps
    for i in range(st):
        # 由于trainset是词频向量，所以只需要将之加入对应的类中，即可
        pn[int(trainlabel[i])] += trainset[i]
        # 所有单词词频之和就是单词总数
        ps[int(trainlabel[i])] += sum(trainset[i])
        print(str(i) + ' is ok')
    # pv 每个类中，每个词出现的概率
    pv = np.zeros((len(num), sw), dtype=float)
    for i in range(len(num)):
        # 用某个单词的词频除以单词总数，即为单词出现的概率，取对数为了将乘法转换为加法
        pv[i] = np.log(pn[i]/ps[i])
    return pv, priori

def naivebayes(testset, testlabel, pv, priori):
    results = []
    correct = 0
    # st sw num含义与trainbayes方法中相同，只是这是测试集
    st = len(testset)
    sw = len(testset[0])
    num = len(priori)
    for i in range(st):
        # 该文档属于某一类的概率
        pn = np.zeros(num, dtype=float)
        result = []
        result.append(i)
        result.append(testlabel[i])
        # 计算该文档属于某一类的后验概率
        for j in range(num):
            # 根据贝叶斯公式，P(A|B)=P(B|A)P(A)/P(B)，由于P(B)是相同的，故而只需要考虑分子，分子中认为每个单词相互独立
            # 转化为对数，将乘法转化为加法
            # p.log(priori[j])即P(A)，先验概率；sum(testset[i]*pv[j])即P(B|A)，这里pv就是每个类中某个次出现的概率，而testset[i]统计了词频，综合考虑了二者
            pn[j] = np.log(priori[j]) + sum(testset[i]*pv[j])
        maxp = 0
        # 找到pn中的最大值，即认定为该文档属于的类别
        for j in range(len(pn)):
            if pn[j] > pn[maxp]:
                maxp = j
        result.append(maxp)
        result.append(str(result[1]) == str(maxp))
        # correct记录计算正确的文档数，除以文档总数作为准确率
        if str(result[1]) == str(maxp):
            correct += 1
        results.append(result)
        print(str(result[1]) == str(maxp))
    return results, correct/len(results)
def main():
    print('NBC')
    dataprocessing(rootpath)
    print("divide labels")
    train, test = filedivide(labels)
    # 返回单词出现的概率和每个类的先验概率
    pv, priori = trainbayes(np.array(trainset), np.array(trainlabel))
    # 返回测试结果和准确率
    bayes, ap = naivebayes(np.array(testset), np.array(testlabel), pv, priori)
    print(bayes)
    print(ap)
if __name__ == '__main__':
    main()