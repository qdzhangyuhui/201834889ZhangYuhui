from collections import Counter
import  random
list=['111','111','aaa','111','aaa','111','aaa','111','aaa','111','aaa','111','aaa','111','aaa','111','aaa','111','aaa']
test = []
train = []
num = Counter(list)
print(num)
print(num.items())
begin = 0
# 统计每一类的文档数，由于读取是按顺序排列，所以设定begin end来标注该类的地一片文档和最后一篇文档
for k, v in num.items():
    print(k)
    print(v/5)