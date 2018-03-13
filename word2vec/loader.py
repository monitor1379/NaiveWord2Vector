# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com

@version: 1.0
@file: loader.py
@time: 18-3-13 下午11:54

这一行开始写关于本文件的说明与解释
"""

from collections import OrderedDict


class Corpus(object):

    def load_monitor1379_data(self, filename):
        # 读取sentences
        self.sentences = []
        with open(filename, 'r') as f:
            for sentence in f.readlines():
                self.sentences.append(sentence[:-1].split(' '))

        # 构造word set
        words_list = set()
        for sentence in self.sentences:
            for word in sentence:
                words_list.add(word)
        words_list = list(words_list)
        words_list.sort()
        self.words_len = len(words_list)

        # 编号
        self.word2id = {}
        self.id2word = {}
        for i, word in enumerate(words_list):
            self.word2id[word] = i
            self.id2word[i] = word



def run():
    corpus = Corpus()
    corpus.load_monitor1379_data('../data/monitor1379-data-100.txt')


if __name__ == '__main__':
    run()
