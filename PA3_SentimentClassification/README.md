# Sentiment Classification

Chinese Sentiment Classification with several deep learning models. Models: MLP、DAN、CNN、RNN、RCNN.

- Embed words
  Run `word2vec.py` script. Requires pre-trained word vectors `sgns.sougou.char`。Generates file`embed_train.npy, embed_test.npy, label_train.npy, label_test.npy` in the directory `data`。
- Run
  Run `main.py` script，two args required: train/test, model_type. Example as follows. 
  Default setting: test CNN

```bash
$python main.py test DAN
====Load Data====
====Load Finish====
====Test Model DAN====
TRAIN SET	acc:70.5332, f1:48.3673, corr:0.6327
DEV SET  	acc:63.1000, f1:34.5579, corr:0.5955
TEST SET 	acc:64.3275, f1:35.3958, corr:0.6045
```

```bash
$python main.py train CNN
====Load Data====
====Load Finish====
====Train Model CNN====
epoch 0
TRAIN	loss:1.6523, acc:42.4228, f1:11.6566, corr:0.3904
DEV  	loss:1.4015, acc:56.3000, f1:16.8337, corr:0.4846
epoch 1
TRAIN	loss:1.3949, acc:53.6015, f1:17.9161, corr:0.4822
DEV  	loss:1.2888, acc:58.7000, f1:18.2565, corr:0.5360
epoch 2
TRAIN	loss:1.2441, acc:58.3723, f1:25.6688, corr:0.5469
DEV  	loss:1.2134, acc:60.7000, f1:24.2522, corr:0.5813
......
```
