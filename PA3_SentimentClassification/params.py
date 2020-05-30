MLP_params = {
    'model_type': 'MLP',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'hidden_size': 512,
    'dropout': 0.5
}
DAN_params = {
    'model_type': 'DAN',
    'lr': 1e-2,
    'weight_decay': 0,
    'num_epochs': 100,
    'hidden_size': 256,
    'dropout': 0.5,
    'word_dropout': 0.5
}
CNN_params = {
    'model_type': 'CNN',
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'num_epochs': 10,
    'num_filters': 128,
    'filter_size': (2, 3, 4),
    'dropout': 0.5
}
RNN_params = {
    'model_type': 'RNN',
    'lr': 1e-3,
    'weight_decay': 0,
    'num_epochs': 50,
    'hidden_size': 256,
    'Attention': True
}
RCNN_params = {
    'model_type': 'RCNN',
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'num_epochs': 20,
    'hidden_size': 128,
    'num_filters': 128
}
