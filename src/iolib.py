# -*_coding:utf8-*-
import pandas as pd
import numpy as np
import math as math
import os as os
import jieba as jieba
import time as time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.model_selection import train_test_split

project_root_path = os.path.abspath('..')
raw_corpus_path = project_root_path + r'/corpus/DMSC.csv/DMSC.csv'
stop_word_path = project_root_path + r"/dict/stop_word.txt"
rnn_model_path = project_root_path + r"/model/keras.model"
douban_comments_file_path = project_root_path + r"/corpus/douban_comments.txt"


def read_stop_words_list(file_path=stop_word_path):
    result_list = []
    with open(file_path, 'r') as f:
        for line in f:
            result_list.append(line)
        return result_list


def read_raw_comment(file_path=raw_corpus_path):
    raw_data_df = pd.read_csv(file_path)
    stop_words_list = read_stop_words_list()
    return_list = []
    cut_total_start = time.clock()
    iter_index = 0
    total_used_time = 0
    iter_start = time.clock()
    raw_data_df = raw_data_df.ix[:100000, :]
    for index, row in raw_data_df.iterrows():
        iter_index += 1
        cut_list = []
        comment = row["Comment"].replace("\r\n", "").replace("\t","")
        score = row["Star"]
        cut_list.append(str(score))
        lcut = jieba.lcut(comment)
        cut_list.extend([x for x in lcut if x not in stop_words_list])
        cut_str = " ".join(cut_list)
        return_list.append(cut_str)
        if iter_index % 10000 == 0:
            elapsed = (time.clock() - iter_start)
            total_used_time += elapsed
            print("Cut the comments %d / %d: %.4f sec, time left estimated:%.4f min" % (
                iter_index, len(raw_data_df), elapsed,
                ((len(raw_data_df) - iter_index) * (total_used_time / iter_index))/60))
            iter_start = time.clock()
    elapsed = (time.clock() - cut_total_start)
    print("Cut all comments complete, time Used: %.4f sec" % elapsed)
    return return_list


if __name__ == "__main__":
    corpus_list = read_raw_comment()
    fl = open(douban_comments_file_path, 'w')
    for i in corpus_list:
        fl.write(i)
        fl.write("\n")
    fl.close()
