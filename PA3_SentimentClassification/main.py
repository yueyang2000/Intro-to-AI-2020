import numpy as np
import sys
import datetime
from copy import deepcopy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# different models
from model import *
# dataset and stat
from data_stat import *
# default params for all models
from params import *

batch_size = 64
print("====Load Data====")
train_set, dev_set, test_set = load_data()
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(
    dev_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False)
print("====Load Finish====")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(params):

    model_type = params['model_type']
    lr = params['lr']
    num_epochs = params['num_epochs']
    weight_decay = params['weight_decay']

    model = None
    if(model_type == 'MLP'):
        model = MLP(params)
    elif(model_type == 'DAN'):
        model = DAN(params)
    elif(model_type == 'CNN'):
        model = CNN(params)
    elif(model_type == 'RNN'):
        model = RNN(params)
    elif(model_type == 'RCNN'):
        model = RCNN(params)
    else:
        print('param error')
        return

    model.to(device)

    print("====Train Model "+model_type+"====")
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0
    best_epoch = -1
    best_model = None

    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()
        train_stat = Stat()
        for batch in train_loader:
            optimizer.zero_grad()
            text, label, class_idx = batch
            pred_label = model(text)

            loss = loss_function(pred_label, class_idx)
            loss.backward()
            optimizer.step()
            train_stat.append(loss.item(), label, pred_label)
        t_loss, t_acc, t_f1, t_corr = train_stat.eval()
        print("TRAIN\tloss:%.4f, acc:%.4f, f1:%.4f, corr:%.4f" % (
            t_loss, t_acc, t_f1, t_corr))

        model.eval()
        dev_stat = Stat()
        with torch.no_grad():
            for batch in dev_loader:
                text, label, class_idx = batch
                pred_label = model(text)
                loss = loss_function(pred_label, class_idx)
                dev_stat.append(loss.item(), label, pred_label)
            d_loss, d_acc, d_f1, d_corr = dev_stat.eval()
            print("DEV  \tloss:%.4f, acc:%.4f, f1:%.4f, corr:%.4f" % (
                d_loss, d_acc, d_f1, d_corr))
            if(d_acc > best_acc):
                best_acc = d_acc
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch

    print("====Train End====")
    print("best acc:%.4f" % (best_acc))

    torch.save(best_model, model_type + '.pkl')
    model.load_state_dict(best_model)
    print("====Test====")
    test_stat = Stat()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            text, label, class_idx = batch
            pred_label = model(text)
            test_stat.append(0, label, pred_label)
        t_loss, t_acc, t_f1, t_corr = test_stat.eval()
        print("TEST\tacc:%.4f, f1:%.4f, corr:%.4f" % (t_acc, t_f1, t_corr))
        return t_acc, t_f1, t_corr


def test_performance(params):
    acc_list = []
    f1_list = []
    corr_list = []
    save = sys.stdout
    print(params)
    for i in range(5):
        print("run test", i)
        sys.stdout = None
        acc, f1, corr = run(params)
        sys.stdout = save
        acc_list.append(acc)
        f1_list.append(f1)
        corr_list.append(corr)
    acc = sum(acc_list) / 5
    f1 = sum(f1_list) / 5
    corr = sum(corr_list) / 5
    print("PERFORMANCE: acc:%.4f, f1:%.4f, corr:%.4f" % (acc, f1, corr))


def test_model(params):
    model_type = params['model_type']
    if(model_type == 'MLP'):
        model = MLP(params)
    elif(model_type == 'DAN'):
        model = DAN(params)
    elif(model_type == 'CNN'):
        model = CNN(params)
    elif(model_type == 'RNN'):
        model = RNN(params)
    elif(model_type == 'RCNN'):
        model = RCNN(params)

    print("====Test Model " + model_type + "====")
    state_path = 'model/' + model_type + '.pkl'
    state = torch.load(state_path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        stat = Stat()
        for batch in train_loader:
            text, label, class_idx = batch
            pred_label = model(text)
            stat.append(0, label, pred_label)
        loss, acc, f1, corr = stat.eval()
        print("TRAIN SET\tacc:%.4f, f1:%.4f, corr:%.4f" % (acc, f1, corr))

        stat = Stat()
        for batch in dev_loader:
            text, label, class_idx = batch
            pred_label = model(text)
            stat.append(0, label, pred_label)
        loss, acc, f1, corr = stat.eval()
        print("DEV SET  \tacc:%.4f, f1:%.4f, corr:%.4f" % (acc, f1, corr))

        stat = Stat()
        for batch in test_loader:
            text, label, class_idx = batch
            pred_label = model(text)
            stat.append(0, label, pred_label)
        loss, acc, f1, corr = stat.eval()
        print("TEST SET \tacc:%.4f, f1:%.4f, corr:%.4f" % (acc, f1, corr))


def main():
    model_type = None
    option = None
    types = ['MLP', 'DAN', 'CNN', 'RNN', 'RCNN']
    options = ['train', 'test']
    if(len(sys.argv) != 3):
        print('arg not given')
        print('Use default setting: test CNN')
        option = 'test'
        model_type = 'CNN'
    else:
        option = sys.argv[1]
        model_type = sys.argv[2]
        if((option not in options) or (model_type not in types)):
            print('arg invalid')
            print('Use default setting: test CNN')
            option = 'test'
            model_type = 'CNN'

    params = None
    if(model_type == 'MLP'):
        params = MLP_params
    elif(model_type == 'DAN'):
        params = DAN_params
    elif(model_type == 'CNN'):
        params = CNN_params
    elif(model_type == 'RNN'):
        params = RNN_params
    elif(model_type == 'RCNN'):
        params = RCNN_params

    if(option == 'test'):
        test_model(params)
        # Use this command to test a model in the model/ directory
        # Print test result on train,dev,test set
    elif(option == 'train'):
        run(params)
        # Use this command to train a model with your param
        # Output the training process

    # test_performance(params)
    # Use this command to run a param setting for 5 times to evaluate performance
    # No output for training process


if __name__ == '__main__':
    main()
