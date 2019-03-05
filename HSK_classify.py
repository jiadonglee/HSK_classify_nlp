# -*- coding: utf-8 -*-
# ''''
# HSK 是面向留学生的汉语水平考试，从简到难分为六个级别(HSK1至HSK6)，
# 该算法实现HSK语料的级别自动判定

# train.txt: 为训练数据，包含 HSK 级别、题型、文本
# test.txt: 为测试数据，包含题型、文本
# HSK.csv: HSK词汇表.xlsx，给出了大纲5000词及其对应的HSK级别，转化为文件HSK.csv便于读取
# stop_word_zh.txt: 中文停用词表
# ''''
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import numpy as np
import textblob, string, jieba, re, os, sys
from keras import layers, models, optimizers

def _read_data(filename):
    # 加载数据集
    data = open(filename).read()
    labels, qs_class, texts = [], [], []
    content_seg = []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])# label
        qs_class.append(content[1])# 题型
        texts.append(re.sub('\W', '', content[2]))# 未分词的文本
        content_temp = jieba.cut(texts[i]) # 分词处理
        content_seg.append(" ".join(content_temp))# 分词存入content_seg
    labels.pop(0) # 删除数据第一行的“级别”,“text”
    content_seg.pop(0)
    return labels, qs_class, texts, content_seg

labels, qs_class, texts, content_seg = _read_data('train.txt')

# =========加载HSK词汇表，作为另一部分的训练数据======
HSK_df = pd.read_csv('HSK.csv')
train_X_add = HSK_df['text']
train_y_add = HSK_df['label'].tolist()
encoder = preprocessing.LabelEncoder()

train_y_add = encoder.fit_transform(train_y_add)

# ===========加载中文停用词表=============
word = open('stop_word_zh.txt').read()
st_word = []
for i, line in enumerate(word.split("\n")):
    content = line.split()
    st_word.append(content[0])
my_stop_words = text.ENGLISH_STOP_WORDS.union(st_word)

#================特征工程========================

def vectorize(stop_words, content_seg, train_X_add, labels):
    # 使用TfidfVectorizer初始化向量空间模型
    vectorizer = TfidfVectorizer(sublinear_tf = True, \
                                decode_error = 'ignore', stop_words = my_stop_words)
    transformer = TfidfTransformer()# 统计每个词语的TF-IDF权值

    # 文本转化为词频矩阵
    content_seg.extend(train_X_add)
    labels.extend(train_y_add)
    content_tdm = vectorizer.fit_transform(content_seg)
    return content_tdm, vectorizer

# content_tdm ,vectorizer = vectorize(my_stop_words, content_seg, train_X_add, labels)

def _split_data(content_tdm, labels):
    # 将数据集分为训练集和验证集
    train_X, test_X, train_y, test_y = model_selection.train_test_split(content_tdm, labels, test_size = 0.2)
    train_y = encoder.fit_transform(train_y) # label编码为目标变量
    test_y = encoder.fit_transform(test_y)
    return train_X, test_X, train_y, test_y
# train_X, test_X, train_y, test_y = _split_data(content_tdm, labels)

def model(train_X, train_y):
    # 训练分类器
    clf = MultinomialNB().fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    return pred_y

def metrics_result(actual, predict):
    # 准确率预测
    print('accuracy:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
#     print('recall:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))

# metrics_result(test_y, pred_y)

def predict_result(filename):
    # 预测结果
    data_ans = open(filename).read()
    qs_ans, texts_out, texts_ans, temp_ans, seg_ans = [], [], [], [], []
    for i, line in enumerate(data_ans.split("\n")):
        content_ans = line.split()
        qs_ans.append(content_ans[0])
        texts_out.append(content_ans[1])
        texts_ans.append(re.sub('\W', '', content_ans[1]))
        temp_ans = jieba.cut(texts_ans[i]) # 分词处理
        seg_ans.append(" ".join(temp_ans))# 分词存入content_seg
    seg_ans.pop(0)
    return seg_ans
seg_ans = predict_result('test.txt')
# ans_tdm = vectorizer.transform(seg_ans)

# ''''
# 以下为主程序，所用函数为vectorize, _split_data, model, metrics_result
# ''''

loops = 20
for i in range(loops):
    content_tdm ,vectorizer = vectorize(my_stop_words, content_seg, train_X_add, labels)
    train_X, test_X, train_y, test_y = _split_data(content_tdm, labels)
    pred_y = model(train_X, train_y)
    metrics_result(test_y, pred_y)