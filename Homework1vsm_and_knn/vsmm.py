# coding=utf-8
import os
import nltk
import re
import string
import math
import pandas as pd

rootpath = '..\\datatest\\dataset'
wordlist = []  # 维度表
vectors = []  # 生产的文档向量集
docs = []  # 存储处理好的文档内容集合
new_wordlist=[]
threshold_value=50
def doc_word_TF(doc):
    doctf = {}
    for word in doc:
        if word not in doctf:
            doctf[word] = 1
        else:
            doctf[word] += 1
    return doctf


def word_IDF(word):
    count = 0
    for doc in docs:
        if word in set(doc):
            count = count + 1
    return math.log(len(docs) / count), count


def creatvectors():
  with open('vector1.csv', 'w') as fi:
    for doc in docs:
        vector = [0.0 for i in range(len(new_wordlist))]
        tflist = doc_word_TF(doc[0:-1])
        for word in doc[0:-1]:
            tf = tflist[word]
            idf, df = word_IDF(word)
            if df > threshold_value:
                vector[new_wordlist.index(word)] = tf * idf
        vector.append(doc[-1])
        fi.write(str(vector) + ',')
        fi.write('\n')
        print(str(vector)+'\n')
        #vectors.append(vector)
  fi.close()



def main():
    # 遍历文件夹进行预处理
    label = 1
    print("data Processing")
    for folderlists in os.listdir(rootpath):
        # print(folderlists)
        path = os.path.join(rootpath, folderlists)
        # print(path)
        for file in os.listdir(path):
            print('current file:'+file)
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
                  if not token.isdigit()and token.isalpha():#去除数字
                    token = stem.stem(token)  # Stemming
                    if token not in stopwords:  # 去停用词
                        doc.append(token)#存储处理好的文章
                        if token not in wordlist:  # 去重
                            wordlist.append(token)
                doc.append(label)#为文章打标签
                docs.append(doc)
        label += 1#一个文件夹为一个标签
    # print(docs)
    # 过滤并输出输出维度表
    print('output wordlist')
    for word in wordlist:#过滤DF小于20的word
        idf, df = word_IDF(word)
        if df > threshold_value:
            new_wordlist.append(word)
    with open('wordlist1.txt', 'w', errors="ignore") as f:
        for w in new_wordlist:
            f.write(w + '\n')
    f.close()
    print('output wordlist finish')
    print('begin to create vectors')
    creatvectors()


if __name__ == '__main__':
    main()

