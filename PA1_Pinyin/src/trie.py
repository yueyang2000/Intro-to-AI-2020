#-*- coding : utf-8 -*-
# coding: utf-8

import pickle
import gc
unigram = dict()
bigram = dict()
trigram = dict()
root = {}
f = open('model_pad.pkl', 'rb')
duoyin = pickle.load(f)
root['cnt'] = duoyin['cnt']
for k, v in duoyin.items():
    # if v == 1:
    #     continue
    if k == 'cnt':
        continue
    if len(k) == 2:
        unigram[k] = v
    elif len(k) == 4:
        bigram[k] = v
    else:
        trigram[k] = v

del duoyin
gc.collect()
print('building trie...')

for k, v in unigram.items():
    root[k] = dict()
    root[k]['f'] = v
print('here')
for k, v in bigram.items():
    root[k[0:2]][k[2:4]] = dict()
    root[k[0:2]][k[2:4]]['f'] = v
print('here')
cnt = 0
for k, v in trigram.items():
    root[k[0:2]][k[2:4]][k[4:6]] = v
    cnt += 1
    if cnt % 100000 == 0:
        print(cnt)


print('here')

with open('trie_sina.pkl', 'wb') as f:
    pickle.dump(root, f, protocol=pickle.HIGHEST_PROTOCOL)
