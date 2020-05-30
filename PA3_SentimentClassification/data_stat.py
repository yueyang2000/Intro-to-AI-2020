import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512
DIM = 300


class myDataset(Dataset):
    def __init__(self, text, label):
        super().__init__()
        self.text = text.to(device)
        self.label = label.to(device)
        self.class_idx = torch.argmax(
            self.label, dim=1, keepdim=False).to(device)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.text[index], self.label[index], self.class_idx[index]


class Stat():
    def __init__(self, ):
        self.loss = []
        self.labels = []
        self.pred_labels = []
        self.classes = []
        self.pred_classes = []

    def append(self, loss, label, pred_label):
        pred_class = pred_label.argmax(dim=1).cpu().detach().numpy()
        true_class = label.argmax(dim=1).cpu().numpy()
        self.loss.append(loss)
        self.labels.extend(label.cpu().detach().numpy())
        self.pred_labels.extend(pred_label.cpu().detach().numpy())
        self.classes.extend(true_class)
        self.pred_classes.extend(pred_class)

    def eval(self):
        loss = sum(self.loss) / len(self.loss)
        acc = accuracy_score(self.classes, self.pred_classes) * 100
        f1 = f1_score(self.classes, self.pred_classes,
                      average='macro') * 100
        corr = sum([pearsonr(self.pred_labels[i], self.labels[i])[0]
                    for i in range(len(self.labels))]) / len(self.labels)
        return loss, acc, f1, corr


def load_data():
    embed_train = np.load('data/embed_train.npy', allow_pickle=True)
    embed_test = np.load('data/embed_test.npy', allow_pickle=True)
    label_train = np.load('data/label_train.npy', allow_pickle=True)
    label_test = np.load('data/label_test.npy', allow_pickle=True)

    label_train_tensor = torch.tensor([np.array(label_train[i]) / np.sum(label_train[i]) for i in range(len(label_train))],
                                      dtype=torch.float)
    label_test_tensor = torch.tensor([np.array(label_test[i]) / np.sum(label_test[i]) for i in range(len(label_test))],
                                     dtype=torch.float)
    label_dev_tensor = label_test_tensor[0:1000]
    label_test_tensor = label_test_tensor[1000:]

    text_train_tensor = torch.zeros(len(embed_train), MAX_LEN, DIM)
    for i, t in enumerate(embed_train):
        t_tensor = torch.tensor(t)
        text_train_tensor[i, 0:len(t), :] = t_tensor
    text_test_tensor = torch.zeros(len(embed_test), MAX_LEN, DIM)
    for i, t in enumerate(embed_test):
        t_tensor = torch.tensor(t)
        text_test_tensor[i, 0:len(t), :] = t_tensor
    text_dev_tensor = text_test_tensor[0:1000]
    text_test_tensor = text_test_tensor[1000:]

    train_set = myDataset(text_train_tensor, label_train_tensor)
    dev_set = myDataset(text_dev_tensor, label_dev_tensor)
    test_set = myDataset(text_test_tensor, label_test_tensor)
    return train_set, dev_set, test_set
    # np.save('data/text_train_tensor.npy', text_train_tensor)
    # np.save('data/label_train_tensor.npy', label_train_tensor)
    # np.save('data/text_dev_tensor.npy', text_dev_tensor)
    # np.save('data/label_dev_tensor.npy', label_dev_tensor)
    # np.save('data/text_test_tensor.npy', text_test_tensor)
    # np.save('data/label_test_tensor.npy', label_test_tensor)
