import torch
import torch.optim as optim
from torchvision import transforms, models
import pandas as pd

from utils.model import Model
import config
from utils.dataloader import Flickr30kDataset
from utils.train_test import initialize_storage, train_model, test_model


# GLOBAL VARIABLES
IMAGES_DIR = './data/images/'
CAPTIONS_DIR = './data/captions_cleaned.csv'
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 0.8, 0.1, 0.1
BATCH_SIZE = 32
NUM_EPOCHS = 20
EMBEDDING_DIM = 256
LEARNING_RATE = config.configs[config.NUM_CONFIG]["LEARNING_RATE"]
CRITERION = config.configs[config.NUM_CONFIG]["CRITERION"]
OPTIMIZER = config.configs[config.NUM_CONFIG]["OPTIMIZER"]
FC_LAYERS = config.configs[config.NUM_CONFIG]["FC_LAYERS"]
ACTIVATIONS = config.configs[config.NUM_CONFIG]["ACTIVATIONS"]
BATCH_NORMS = config.configs[config.NUM_CONFIG]["BATCH_NORMS"]


initialize_storage()

# Data transfpormation
transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])

# Import dataset
captions_df = pd.read_csv(CAPTIONS_DIR, sep='|', header=None, names=['image_id', 'comment_number', 'caption'], skipinitialspace=True)
dataset = Flickr30kDataset(captions_df, IMAGES_DIR, transform)

# Configure models
models = [models.densenet201, models.vgg16, models.resnet50]
names = ["densenet201","vgg16","resnet50"]

# Split in train, validation and test


# Train and validation loop
for model_function, model_name in zip(models, names):
    model = Model()
    for epoch in range(NUM_EPOCHS):
        train_model(model, train_loader, OPTIMIZER, CRITERION, epoch)
        test_model(model, val_loader, OPTIMIZER, CRITERION, epoch, type="val")

# Save trained model
#torch.save(model.state_dict(), f'utils/saved_models/{model.name}_config{config.NUM_CONFIG}.pth')

# Test
