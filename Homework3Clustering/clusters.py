import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score


# 加载文档向量
def loaddata(filepath):
    vectors = []
    label = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            vector = []
            item = str(line).split(',')
            # 每一行前n-1个元素是tfidf向量
            for i in range(len(item) - 1):
                vector.append(float(item[i]))
            # 最后一位是标签
            label.append(str(item[len(item) - 1]).replace('\n', ''))
            vectors.append(vector)
            line = f.readline()
    return vectors, label


# K-Means
def kmeans(X, y, k):
    # 适配
    km = KMeans(n_clusters=k).fit(X)
    # 预测样本标签
    pr = km.predict(X)
    # 计算NMI（Normalized Mutual Information）归一化互信息
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# AffinityPropagation
def affinitypropagation(X, y):
    ap = AffinityPropagation().fit(X)
    pr = ap.predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# Mean-shift
def meanshift(X, y):
    ms = MeanShift().fit(X)
    pr = ms.predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# Spectral clustering
def spectral(X, y, k):
    sc = SpectralClustering(n_clusters=k).fit(X)
    pr = sc.fit_predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# Ward hierarchical clustering
def hierarchical(X, y, k):
    hc = AgglomerativeClustering(n_clusters=k).fit(X)
    pr = hc.fit_predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# DBSCAN
def dbscan(X, y):
    db = DBSCAN().fit(X)
    pr = db.fit_predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


# Gaussian mixtures
def gaussian(X, y, k):
    gm = GaussianMixture(n_components=k).fit(X)
    pr = gm.predict(X)
    nmi = normalized_mutual_info_score(y, pr)
    return nmi


def main():
    filepath = 'data/tfidf.csv'
    vectors, label = loaddata(filepath)
    X = np.array(vectors, dtype=float)
    y = np.array(label, dtype=int)
    # k值选取类别总数
    k = len(Counter(label))
    k_nmi = kmeans(X, y, k)
    a_nmi = affinitypropagation(X, y)
    m_nmi = meanshift(X, y)
    s_nmi = spectral(X, y, k)
    h_nmi = hierarchical(X, y, k)
    d_nmi = dbscan(X, y)
    g_nmi = gaussian(X, y, k)

    print('K-Means:', k_nmi)
    print('AffinityPropagation:', a_nmi)
    print('Mean-shift:', m_nmi)
    print('Spectral clustering:', s_nmi)
    print('Ward hierarchical clustering:', h_nmi)
    print('DBSCAN:', d_nmi)
    print('Gaussian mixtures:', g_nmi)


if __name__ == "__main__":
    main()
