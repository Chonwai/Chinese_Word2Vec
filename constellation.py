import sys
from gensim.models import Word2Vec

modelPath = sys.argv[1]
word1 = sys.argv[2]
word2 = sys.argv[3]

model = Word2Vec.load(modelPath)

res = model.wv.most_similar([word1, word2, '特點'], ['星座', '座'], topn=20)

print('同「' + sys.argv[2] + sys.argv[3] + '」有關嘅嘢喺：')
for item in res:
    print(item[0]+","+str(item[1]))
