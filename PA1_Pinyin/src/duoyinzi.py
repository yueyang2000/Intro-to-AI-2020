#!/usr/bin/env python
import pickle
from pypinyin import lazy_pinyin
import json
import re


def load_pinyin():
    with open('pinyin_dict.pkl', 'rb') as f:
        pinyin_dict = pickle.load(f)
    with open('duoyin_dict.pkl', 'rb') as f:
        duoyin_dict = pickle.load(f)
    return pinyin_dict, duoyin_dict


def feed_data(freq, data, duoyin_dict):
    sentences = re.findall('[\u4E00-\u9FFF]+', data)
    for s in sentences:
        l = len(s)
        if l < 5:
            continue
        else:
            freq['cnt'] += l

        pinyin = lazy_pinyin(s)  # 注音
        zi_yin = []
        for i in range(len(s)):
            try:
                idx = duoyin_dict[s[i]].index(pinyin[i])
            except:
                idx = 0
            zi_yin.append(s[i] + str(idx))

        for i in range(len(s)):

            try:
                freq[zi_yin[i]] += 1
            except:
                freq[zi_yin[i]] = 1
            if i + 1 < len(s):
                try:
                    freq[zi_yin[i] + zi_yin[i + 1]] += 1
                except:
                    freq[zi_yin[i] + zi_yin[i + 1]] = 1
            if i + 2 < len(s):
                try:
                    freq[zi_yin[i] + zi_yin[i + 1] + zi_yin[i + 2]] += 1
                except:
                    freq[zi_yin[i] + zi_yin[i + 1] + zi_yin[i + 2]] = 1


def load_data(freq, path, duoyin_dict):
    with open(path, 'r', encoding='gbk') as f:
        cnt = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            data = json.loads(line)
            feed_data(freq, data['html'], duoyin_dict)
            cnt += 1
            # print('feed {0}th piece of data'.format(cnt))


pinyin_dict, duoyin_dict = load_pinyin()

freq = dict()
freq['cnt'] = 0
month = ['02', '04', '05', '06', '07', '08', '09', '10', '11']
for m in month:
    path = '/Users/yueyang/yiqunyang/大二下/人工智能导论/拼音输入法作业/sina_news_gbk/2016-' + m + '.txt'
    load_data(freq, path, duoyin_dict)
    print(m, 'finished')
with open('duoyin.pkl', 'wb') as f:
    pickle.dump(freq, f, True)
