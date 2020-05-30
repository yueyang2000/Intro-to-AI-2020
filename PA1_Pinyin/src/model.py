import re
from math import exp
import numpy as np
import pickle
import os
import sys
import time


class myModel:
    def __init__(self, lamb=0.99, gamma=100):
        with open('../model/model.pkl', 'rb') as f:
            self.trie = pickle.load(f)
            self.cnt = self.trie['cnt']
        self.gamma = gamma
        self.lamb = lamb

    def get_freq(self, data):
        # 获得字串data的出现次数
        freq = None
        if len(data) == 2:
            d = self.trie.get(data)
            if d:
                freq = d['f']
        elif len(data) == 4:
            d1 = self.trie.get(data[0:2])
            if d1:
                d2 = d1.get(data[2:4])
                if d2:
                    freq = d2['f']
        else:
            d1 = self.trie.get(data[0:2])
            if d1:
                d2 = d1.get(data[2:4])
                if d2:
                    freq = d2.get(data[4:6])
        if freq:
            return freq
        else:
            return 0

    def get_prob1(self, s):
        # 获得一元概率，返回numpy向量
        ans = np.zeros(len(s))
        F1 = np.zeros_like(ans)
        for i in range(len(s)):
            F1[i] = self.get_freq(s[i])
        ans = F1 / self.cnt
        mask = (F1 == 0)
        ans += mask * 1e-100
        ans = np.log(ans)
        return ans

    def get_prob2(self, s1, s2):
        # 获得二元概率，返回numpy矩阵
        l1 = len(s1)
        l2 = len(s2)

        ans = np.zeros((l1, l2))
        # ans[i][j] = logp(s1[i]|s2[j])
        F12 = np.zeros_like(ans)
        F2 = np.zeros_like(ans)
        F1 = np.zeros_like(ans)
        for i in range(l1):
            for j in range(l2):
                c1 = s1[i]
                c2 = s2[j]
                c = c2 + c1
                F12[i][j] = self.get_freq(c)
                F2[i][j] = self.get_freq(c2)
                F1[i][j] = self.get_freq(c1)

        # p(c1|c2) = lamb * p(c2c1|c2) + (1 - lamb) * p(c1)
        mask = (F2 == 0)  # a mask
        F2 += mask  # in case divide by 0
        ans = self.lamb * F12 / F2
        ans = ans * (1 - mask)  # where F2=0, set to zero

        ans += (1 - self.lamb) * F1 / self.cnt
        ans += 1e-200 * (ans == 0)  # something really small
        ans = np.log(ans)  # compute the log
        return ans

    def get_prob3(self, s1, s2, s3):
        # 获得三元概率，返回numpy矩阵
        gamma = self.gamma
        l1 = len(s1)
        l2 = len(s2)

        ans = np.zeros((l1, l2))
        # ans[i][j] = logp(s1[i]|s2[j]s3[j])
        F123 = np.zeros_like(ans)
        F23 = np.zeros_like(ans)
        F12 = np.zeros_like(ans)
        F2 = np.zeros_like(ans)
        F1 = np.zeros_like(ans)
        for i in range(l1):
            for j in range(l2):
                c1 = s1[i]
                c2 = s2[j]
                c3 = s3[j]
                c123 = c3 + c2 + c1
                c23 = c3 + c2
                c12 = c2 + c1
                F123[i][j] = self.get_freq(c123)
                F23[i][j] = self.get_freq(c23)
                F12[i][j] = self.get_freq(c12)
                F2[i][j] = self.get_freq(c2)
                F1[i][j] = self.get_freq(c1)

        p1 = F123 / (F23 + (F123 == 0))  # in case F123=0
        p2 = F12 / (F2 + (F12 == 0))  # in case F12=0
        p3 = F1 / self.cnt

        lamb1 = F23 / (F23 + gamma)
        lamb2 = (1 - lamb1) * F2 / (F2 + gamma)
        ans = lamb1 * p1 + lamb2 * p2 + (1 - lamb1 - lamb2) * p3
        # p(c1|c2c3) = lambda1*p(c1|c2c3) + lambda2*p(c1|c2) +(1-lambda1-lambda2)*p(c1)

        ans += 1e-200 * (ans == 0)  # something really small
        ans = np.log(ans)  # compute the log
        return ans


def load_pinyin():
    with open('../model/pinyin_dict.pkl', 'rb') as f:
        pinyin_dict = pickle.load(f)
    with open('../model/duoyin_dict.pkl', 'rb') as f:
        duoyin_dict = pickle.load(f)
    return pinyin_dict, duoyin_dict


def convert(data, model, pinyin_dict, duoyin_dict, gram, k, pad=True, debug=False):
    # 拼音汉字转换器
    # input:
    #    data: 输入的拼音串
    #    model: 词频模型
    #    pinyin_dict, duoyin_dict: 拼音词典，多音字词典
    #    gram: 二元或三元模型
    #    k: 维特比算法记忆前k种种选择
    #    pad: 是否在句首句尾添加st和ed标记
    #    debug: 调试输出前5种选择
    # output：
    #    转换完毕的中文字串

    if gram == 2:
        k = 1  # 二元模型只需要贪心动归
    pinyin = data.split()
    if pad:
        n = len(pinyin) + gram
        if gram == 3:
            grid = [['st'], ['st']]  # padding
        else:
            grid = [['st']]
    else:
        n = len(pinyin)
        grid = []
    # grid[i][j] is the jth possibility of the ith character

    for p in pinyin:
        try:
            li = pinyin_dict[p].copy()
            for i in range(len(li)):
                try:
                    idx = duoyin_dict[li[i]].index(p)
                except:
                    idx = 0
                li[i] = li[i] + str(idx)  # zi_yin
            grid.append(li)
        except:
            return None
    if pad:
        grid.append(['ed'])

    choice = grid[0]
    prev_prob = model.get_prob1(grid[0])
    for i in range(n - 1):
        cur_k = min(len(choice), k)
        prob_matrix = None
        l = len(grid[i + 1])

        # 计算转移矩阵
        if i == 0:
            prob_matrix = model.get_prob2(grid[1], grid[0])
        else:
            if gram == 2:
                s2 = list()
                for c in choice:
                    s2.append(c[-2:])
                prob_matrix = model.get_prob2(grid[i + 1], s2)
            else:  # gram == 3
                s2 = list()
                s3 = list()
                for c in choice:
                    s2.append(c[-2:])
                    s3.append(c[-4:-2])
                prob_matrix = model.get_prob3(
                    grid[i + 1], s2, s3)

        # 实现维特比算法
        prob = prob_matrix + prev_prob
        prev_prob = np.zeros(l * cur_k)
        top_k = np.zeros(l * cur_k)
        for j in range(cur_k):
            prev_prob[j * l:(j + 1) * l] = np.max(prob, axis=1)
            top = np.argmax(prob, axis=1)
            top_k[j * l: (j + 1) * l] = top
            for t in range(top.shape[0]):
                prob[t][top[t]] = float('-inf')
                # 所有字的 top k 挨在一起

        new_choice = []
        for j in range(cur_k):
            for z1 in range(l):
                new_choice.append(
                    choice[int(top_k[j * l + z1])] + grid[i + 1][z1])
        choice = new_choice
    try:
        idx = np.argmax(prev_prob)
        ans = choice[idx][::2]

        if debug:
            tmp = min(len(choice), 5)
            print('top', tmp)
            for i in range(tmp):
                print(choice[idx][::2], prev_prob[idx])
                prev_prob[idx] = float('-inf')
                idx = np.argmax(prev_prob)

        if pad:
            if gram == 3:
                ans = ans[2:-1]
            else:
                ans = ans[1:-1]
        return ans
    except:
        return None


def convert_file(model, pinyin_dict, duoyin_dict, gram,k, input, output):
    fin = open(input,'r')
    fout = open(output,'w')
    start = time.time()
    first = True
    while True:
        line = fin.readline()
        pinyin = line.strip('\n')
        pinyin = pinyin.lower()
        if len(pinyin) == 0:
            break;
        try:
            ans = convert(pinyin, model, pinyin_dict, duoyin_dict, gram, k, pad=True)
            if first:
                fout.write(ans)
                first = False
            else:
                fout.write('\n' + ans)
        except:
            fout.write('\n')
    fin.close()
    fout.close()
    finish = time.time()
    print('Convert finish!','Elapsed:',finish - start, 's')

    


def test(model, pinyin_dict, duoyin_dict, gram, k, pad=True):
    ans = []
    with open('../data/input.txt', 'r') as fin:
        time_start = time.time()
        while True:
            line = fin.readline()
            if len(line) == 0:
                break
            pinyin = line.strip('\n')
            try:
                a = convert(pinyin, model,
                            pinyin_dict, duoyin_dict, gram=gram, k=k, pad=pad)
            except:
                a = None
                print('error:', pinyin)
            ans.append(a)
        time_end = time.time()
        print('convert finished, total time:', time_end - time_start, 's')
    with open('../data/answer.txt', 'r') as fans:
        real_ans = []
        while True:
            line = fans.readline()
            if len(line) == 0:
                break
            real_ans.append(line.strip('\n'))
        cnt = 0
        fout = open('negative.txt', 'w')
        zi_cnt = 0
        zi_total = 0
        for i in range(len(ans)):
            if not ans[i]:
                continue
            zi_total += len(ans[i])
            if(real_ans[i] == ans[i]):
                zi_cnt += len(ans[i])
                cnt += 1
            else:
                for j in range(len(ans[i])):
                    if ans[i][j] == real_ans[i][j]:
                        zi_cnt += 1
                if i < 1000:
                    fout.write(real_ans[i] + '\n')
                    fout.write(ans[i] + '\n\n')
        print('correct:', cnt, 'of', len(ans),
              'accuracy:', 1.0 * cnt / len(ans), 'zi_accuracy:', 1.0 * zi_cnt / zi_total)
