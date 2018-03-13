# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com

@version: 1.0
@file: models.py
@time: 18-3-14 上午12:05

这一行开始写关于本文件的说明与解释
"""

import matplotlib.pyplot as plt
import numpy as np

from word2vec.loader import Corpus


def lookup_row(table, row):
    return table[row: row + 1, :]


def set_row(table, row, value):
    table[row: row + 1, :] = value


def lookup_col(table, col):
    return table[:, col: col + 1]


def softmax(logits):
    shift_logits = logits - np.max(logits)
    exp_shift_logits = np.exp(shift_logits)
    return exp_shift_logits / np.sum(exp_shift_logits)


def sigmoid(z):
    if z > 6:
        return np.array([[1.0]])
    elif z < -6:
        return np.array([[0.0]])
    else:
        return 1 / (1 + np.exp(-z))


def sampling(data):
    return data[np.random.randint(len(data))]


class SkipGram(object):

    def __init__(self, sentences, word2id, id2word, words_len):
        self.sentences = sentences
        self.word2id = word2id
        self.id2word = id2word
        self.words_len = words_len
        self.words_list = list(word2id.keys())

        self.feature_dim = 0
        self.word_vec = None

        self.w0 = None
        self.w1 = None

        self.logger = {}

    def _train_one_step(self, target_word, another_word, label, lr):
        w0 = self.w0
        w1 = self.w1
        x_id = self.word2id[target_word]
        y_id = self.word2id[another_word]
        # TODO(monitor1379): 增加negative sampling
        # 前向计算
        logits = lookup_row(w0, x_id).dot(lookup_col(w1, y_id))
        output = sigmoid(logits)
        loss = np.abs(label - output[0][0])
        print(label, output[0][0])
        self.logger['loss'].append(loss)

        # 反向传播
        d_logit = (label - output) * output * (1 - output)  # TODO(monitor1379): fix
        d_w1 = lookup_row(w0, x_id).T.dot(d_logit)
        d_w0 = d_logit.dot(lookup_col(w1, y_id).T)

        # 更新模型
        self.w1[:, y_id] += d_w1[:, 0] * lr
        self.w0[x_id, :] += d_w0[0, :] * lr

    def train(self, feature_dim, context_window_size, n_epochs, lr):
        self.logger['loss'] = []
        self.w0 = np.random.uniform(low=-0.5 / feature_dim, high=0.5 / feature_dim, size=[self.words_len, feature_dim])
        self.w1 = np.random.uniform(low=-0.5 / feature_dim, high=0.5 / feature_dim, size=[feature_dim, self.words_len])
        random_negative_sampling_times = 3

        for epoch in range(1, n_epochs + 1):
            print('{:0>3}/{}'.format(epoch, n_epochs))
            for sentence in self.sentences:  # 对于每句话
                for target_word_index, target_word in enumerate(sentence):  # 对于每个中心词
                    window_start = max(target_word_index - context_window_size, 0)
                    window_end = min(target_word_index + context_window_size + 1, len(sentence))

                    for context_word_index in range(window_start, window_end):  # 对于每个上下文窗口中的词
                        context_word = sentence[context_word_index]
                        self._train_one_step(target_word, context_word, label=1, lr=lr)
                        for i in range(random_negative_sampling_times):
                            self._train_one_step(target_word, sampling(self.words_list), label=0, lr=lr)

        plt.ylim([0, 1])
        plt.plot(self.logger['loss'])
        plt.show()
        # ***********************************************
        # TODO(monitor1379): Softmax + CrossEntropy
        # ***********************************************
        # logits = x.dot(w1)
        # output = softmax(logits)
        # loss = np.abs(1 - lookup_col(output, y_id))
        # print(lookup_col(output, y_id))
        # print(w0)
        # 反向传播，损失函数是cross entropy
        # d_logit = lookup_col(output, y_id) - 1
        # d_x = d_logit.dot(lookup_col(w1, y_id).T)
        # d_w1 = x.T.dot(d_logit)
        # w0[x_id] += d_x[0, :] * lr
        # w1[:, y_id] += d_w1[:, 0] * lr

    def find_nearest(self, word):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


def run():
    corpus = Corpus()
    corpus.load_monitor1379_data('../data/monitor1379-data-100.txt')
    model = SkipGram(corpus.sentences, corpus.word2id, corpus.id2word, corpus.words_len)
    model.train(feature_dim=150, context_window_size=4, n_epochs=100, lr=0.01)


if __name__ == '__main__':
    run()
