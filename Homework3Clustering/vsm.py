import json
import math


# 将文档转换为向量
def loadfile(filepath):
    # vectors 存放每篇文档的所有单词，是一个二维数组
    vectors = []
    # label 存放每篇文档的标签
    label = []
    with open(filepath, 'r') as f:
        # 读取Tweets.json并作为字典存储在tweets中
        tweets = json.load(f)
    for tweet in tweets:
        # text标签中存放了所有单词，并以空格分隔
        vector = tweet['text'].split(' ')
        vectors.append(vector)
        # cluster标签存放了label
        label.append(str(tweet['cluster']))
    return vectors, label


# 生成单词表
def ctable(vectors):
    # 为了避免添加重复单词，采用集合存储
    wordtable = set([])
    for i in vectors:
        # 对集合求并集
        wordtable = wordtable | set(i)
    # 将集合转换为数组
    wordtable = list(wordtable)
    return wordtable


# 计算tf-idf
def tfidf(doc, vectors, wordtable):
    tfidfvector = []
    # 初始化tf idf为0
    tf = 0
    idf = 0
    # 计算每篇文档的所有单词的词频
    freq = termfreq(doc)
    for i in range(len(wordtable)):
        tfidfvector.append(0)
    for i in set(doc):
        if i not in wordtable:
            continue
        # 取词频作为tf
        tf = freq[i]
        # idf = log((N+1)/(df+1))
        idf = math.log((len(vectors) + 1) / (df(i, vectors) + 1))
        tfidf = tf*idf
        tfidfvector[wordtable.index(i)] = tfidf
    return tfidfvector


# 词频
def termfreq(doc):
    freq = {}
    for i in doc:
        # 每从文档中取出一个单词，词频+1
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    return freq


# 包含某一单词的文档数
def df(word, vectors):
    count = 0
    for i in vectors:
        if word in set(i):
            count += 1
    return count


def main():
    filepath = 'Tweets.json'
    # 加载文档向量和标签
    vectors, label = loadfile(filepath)
    # 生成单词表
    wordtable = ctable(vectors)
    with open('wordtable.txt', 'w') as f:
        for i in wordtable:
            f.write(str(i)+'\n')
    print("save wordtable done.")
    with open('tfidf.csv', 'w') as f:
        f.write("")
    for doc in vectors:
        # 计算每篇文档的tfidf，并连同标签存放在tfidf.csv中
        tfidfvector = tfidf(doc, vectors, wordtable)
        with open('tfidf.csv', 'a') as f:
            for j in tfidfvector:
                f.write(str(j) + ',')
            f.write(label[vectors.index(doc)]+'\n')
        print(vectors.index(doc), ' is done.')
    print("all done.")


if __name__ == '__main__':
    main()
