import multiprocessing
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

segmentationPath = sys.argv[1]

print('Use ' + str(multiprocessing.cpu_count()) + ' CPUs')
print("Loading the config...")
# path = './dataset/wiki_seg_DX.txt'
output1 = './model/wiki_model'
# output2 = './vector/wiki_vector'

print("Training the Model...")
model = Word2Vec(LineSentence(segmentationPath), size=1000, window=7,
                 min_count=2, workers=multiprocessing.cpu_count())
model.save(output1)
# model.wv.save_word2vec_format(output2, binary=False)
