"""
Number of CNN layers: (by changing modules) VGG16-19, ResNet18-34-50
● Number of RNN layers: default 1
● Type of RNN cells: LSTM vs GRU
● Prediction method: character-level, word-level, wordpiece-level
● Optimizer: SGD, Adam, AdamW, …
● Learning rate, learning rate decay, early stopping
"""
configs = [
    {
        # Configuration 0
        "LEARNING_RATE": 0.001,
        "CRITERION": "CrossEntropyLoss",  
        "OPTIMIZER": "Adam",  
        "FC_LAYERS": [512, 256, 128],  
        "ACTIVATIONS": ["ReLU", "ReLU", "ReLU"],  
        "BATCH_NORMS": [True, True, False]  
    },
    {
        # Configuration 1
        "LEARNING_RATE": 0.0005,
        "CRITERION": "MSELoss",  
        "OPTIMIZER": "SGD",  
        "FC_LAYERS": [1024, 512],
        "ACTIVATIONS": ["ReLU", "Tanh"], 
        "BATCH_NORMS": [True, False]  
    },
    {
        # Configuration 2
        "LEARNING_RATE": 0.01,  
        "CRITERION": "CrossEntropyLoss", 
        "OPTIMIZER": "RMSprop",  
        "FC_LAYERS": [2048, 1024, 512],  
        "ACTIVATIONS": ["LeakyReLU", "LeakyReLU", "ReLU"],  
        "BATCH_NORMS": [True, True, True]  
    },
    {
        # Configuration 3
        "LEARNING_RATE": 0.0001,
        "CRITERION": "BCELoss",  
        "OPTIMIZER": "AdamW",  
        "FC_LAYERS": [256, 128],
        "ACTIVATIONS": ["ReLU", "Sigmoid"],  
        "BATCH_NORMS": [False, True]  
    },
    {
        # Configuration 4
        "LEARNING_RATE": 0.005,
        "CRITERION": "MSELoss",  
        "OPTIMIZER": "Adagrad",
        "FC_LAYERS": [512, 256],
        "ACTIVATIONS": ["Tanh", "ReLU"],  
        "BATCH_NORMS": [True, True]  
    },
    {
        # Configuration 5
        "LEARNING_RATE": 0.002,
        "CRITERION": "CrossEntropyLoss", 
        "OPTIMIZER": "Adam",  
        "FC_LAYERS": [512, 512, 256],
        "ACTIVATIONS": ["ReLU", "LeakyReLU", "ReLU"], 
        "BATCH_NORMS": [True, True, True]  
    }
]


