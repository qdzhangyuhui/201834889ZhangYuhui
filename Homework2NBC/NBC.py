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
labels = []
countlist = {}
threshold_value_low = 30
threshold_value_high = 800
def dataprocessing(rootpath):
    print('input data')
    label = 1
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
        print(word, 'cf=', cf)
        if cf > threshold_value_low and cf < threshold_value_high:
            dict.append(word)
    with open('dict.txt', 'w', errors="ignore") as f:
        for w in dict:
            f.write(w + '\n')
    f.close()
    print('output wordlist finish')
def train_naivebyes():
    print()
def main():
    print('NBC')
    dataprocessing(rootpath)
if __name__ == '__main__':
    main()