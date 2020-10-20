import sys
from gensim.models import Word2Vec

modelPath = sys.argv[1]
word = sys.argv[2]

model = Word2Vec.load(modelPath)

res = model.wv.most_similar(word, topn=10)

print('同「' + sys.argv[2] + '」有關嘅嘢喺：')
for item in res:
    print(item[0]+","+str(item[1]))
