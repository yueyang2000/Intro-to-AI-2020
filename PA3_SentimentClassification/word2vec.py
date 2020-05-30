import numpy as np
import torch


def read_text(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        text_list = []
        label_list = []
        cnt = 0
        for line in lines:
            if(len(line) == 0):
                break

            data = line.split('\t')
            label_data = data[1].split(' ')
            label = np.zeros(8)
            for i in range(8):
                label[i] = int(label_data[i + 1].split(':')[1])

            max_score = max(label)
            max_label = None
            abandon = False
            for i, l in enumerate(label):
                if(l == max_score):
                    if(not max_label):
                        max_label = i
                        #print('max_label:', i)
                    else:
                        abandon = True  # 无最大标签
                        break
            if(abandon):
                continue

            text = data[2].strip('\n').split(' ')
            # print(text)
            text_list.append(text)
            label_list.append(label)
            cnt += 1

        print('Total:', cnt)
        return text_list, label_list


def read_vector(num):
    with open('data/sgns.sogou.char', encoding='utf=8') as f:
        line = f.readline()
        dim = line.rstrip().split(' ')[1]
        print('dim:', dim)  # 300
        cnt = 0
        word_vec = {}
        while(True):
            line = f.readline()
            if(len(line) == 0):
                break
            data = line.rstrip().split(' ')

            # 去除非中文字符
            if '\u4e00' > data[0] or data[0] > '\u9fa5':
                continue
            cnt += 1
            word_vec[data[0]] = np.asarray([float(k) for k in data[1:]])
            if(cnt > num):
                break
        return word_vec


def embed_word(word_vec, text_list):
    embed_list = []
    in_cnt = 0
    miss_cnt = 0
    for text in text_list:
        embed = []
        word_cnt = 0
        for word in text:
            if '\u4e00' > word[0] or word[0] > '\u9fa5':
                continue
            word_cnt += 1
            if word in word_vec:
                embed.append(word_vec[word])
                in_cnt += 1
            else:
                embed.append(np.zeros(300))
                miss_cnt += 1
            if(word_cnt == 512):  # choose fisrt 512 words
                break
        embed_list.append(embed)
    print('in_cnt:', in_cnt)
    print('miss_cnt:', miss_cnt)
    return embed_list


def main():
    print('read vectors')
    word_vec = read_vector(100000)
    print('read data')
    text_train, label_train = read_text('data/sinanews.train')
    text_test, label_test = read_text('data/sinanews.test')
    print('embed word')
    embed_train = embed_word(word_vec, text_train)
    embed_test = embed_word(word_vec, text_test)
    np.save('data/embed_train.npy', embed_train, allow_pickle=True)
    np.save('data/embed_test.npy', embed_test, allow_pickle=True)
    np.save('data/label_train', label_train, allow_pickle=True)
    np.save('data/label_test', label_test, allow_pickle=True)
    print('done')


if __name__ == '__main__':
    main()
