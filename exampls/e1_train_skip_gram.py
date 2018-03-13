# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com

@version: 1.0
@file: e1_train_skip_gra m.py
@time: 18-3-14 上午2:39

这一行开始写关于本文件的说明与解释
"""

from word2vec.loader import Corpus
from word2vec.models import SkipGram


def run():
    corpus = Corpus()
    corpus.load_monitor1379_data('../data/monitor1379-data-100.txt')
    model = SkipGram(corpus.sentences, corpus.word2id, corpus.id2word, corpus.words_len)
    model.train(feature_dim=150, context_window_size=4, n_epochs=100, lr=0.01)


if __name__ == '__main__':
    run()
