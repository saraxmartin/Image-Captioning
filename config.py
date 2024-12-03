import torch.nn as nn
import torch.optim as optim

NUM_CONFIG = 2

configs = {
    1:  # Configuration 1
        {
        "LEARNING_RATE": 0.001,
        "CRITERION": nn.CrossEntropyLoss,  
        "OPTIMIZER": optim.Adam, 
        "FC_LAYERS": [512, 256, 128],
        "ACTIVATIONS": ["ReLU", "ReLU", "ReLU"],
        "BATCH_NORMS": [True, True, False]
    },
    2:  # Configuration 2
        {
        "LEARNING_RATE": 0.00005,
        "CRITERION": nn.MSELoss,
        "OPTIMIZER": optim.SGD,
        "FC_LAYERS": [1024, 512],
        "ACTIVATIONS": ["ReLU", "Tanh"],
        "BATCH_NORMS": [True, False]
    },

    3:  # Configuration 3
        {
        "LEARNING_RATE": 0.01,
        "CRITERION": nn.CrossEntropyLoss,
        "OPTIMIZER": optim.RMSprop,
        "FC_LAYERS": [2048, 1024, 512],
        "ACTIVATIONS": ["LeakyReLU", "LeakyReLU", "ReLU"],
        "BATCH_NORMS": [True, True, True]
    },
    4:  # Configuration 4
        {
        "LEARNING_RATE": 0.0001,
        "CRITERION": nn.BCELoss,
        "OPTIMIZER": optim.AdamW,
        "FC_LAYERS": [256, 128],
        "ACTIVATIONS": ["ReLU", "Sigmoid"],
        "BATCH_NORMS": [False, True]
    },
    5:  # Configuration 5
        {
        "LEARNING_RATE": 0.005,
        "CRITERION": nn.MSELoss,
        "OPTIMIZER": optim.Adagrad,
        "FC_LAYERS": [512, 256],
        "ACTIVATIONS": ["Tanh", "ReLU"],
        "BATCH_NORMS": [True, True]
    },
    6:  # Configuration 6
        {
        "LEARNING_RATE": 0.002,
        "CRITERION": nn.CrossEntropyLoss,
        "OPTIMIZER": optim.Adam,
        "FC_LAYERS": [512, 512, 256],
        "ACTIVATIONS": ["ReLU", "LeakyReLU", "ReLU"],
        "BATCH_NORMS": [True, True, True]
    }
}

NUM_CONFIG = 1
