import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
import os
import csv

import config
from utils.dataset import FoodDataset
from utils.train_test import initialize_storage, train_model, test_model
from utils.model import CaptioningModel

# GLOBAL VARIABLES
IMAGES_DIR = 'data/images/'
#IMAGES_DIR = "/export/fhome/vlia03/Image_Captioning/data/images"
CAPTIONS_DIR = "data/info.csv"
#CAPTIONS_DIR = "/export/fhome/vlia03/Image_Captioning/data/info.csv"
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 0.8, 0.1, 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 20
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
ATTENTION_DIM = 256
selected_config = config.configs[config.NUM_CONFIG]
FC_LAYERS = selected_config["FC_LAYERS"]
ACTIVATIONS = selected_config["ACTIVATIONS"]
BATCH_NORMS = selected_config["BATCH_NORMS"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


initialize_storage()

# Data transfpormation
transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])

# Import dataset
info_df = pd.read_csv(CAPTIONS_DIR)
dataset = FoodDataset(info_df, IMAGES_DIR, transform)
VOCAB_SIZE = len(dataset.data_properties["lexicon"])
#print("ORIGINAL VOCAB SIZE", VOCAB_SIZE)
#statistics = dataset.get_statistics(printed=True)
#print(f"Number of images in total: {len(dataset)}")

# Split in train, validation and test
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)
#train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
#print(f"Number of images in train dataset: {len(train_dataset)}")
#print(f"Number of images in validation dataset: {len(val_dataset)}")
#print(f"Number of images in test dataset: {len(test_dataset)}")

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Configure models
name_models = [models.vgg16,models.densenet201, models.resnet50]
names = ["vgg16","densenet201","resnet50"]
name_models = [models.resnet50]
names = ["resnet50"]


VOCAB_SIZE = len(dataset.word2idx)
print("VOCAB SIZE:", VOCAB_SIZE)
gt = dataset.idx2word
#print("NEW VOCAB SIZE", VOCAB_SIZE)
# Train and validation loop
for model_function, model_name in zip(name_models, names):
    model = CaptioningModel(model_function, model_name, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, ATTENTION_DIM)
    model = model.to(DEVICE)
    CRITERION = selected_config["CRITERION"]()  # Loss function
    OPTIMIZER = selected_config["OPTIMIZER"](model.parameters(), lr=selected_config["LEARNING_RATE"])
    SCHEDULER = ExponentialLR(OPTIMIZER, gamma=0.95)  # Reduce LR by 5% per epoch
    
    for epoch in range(NUM_EPOCHS):
        current_lr = SCHEDULER.get_last_lr()[0]  # Get the current learning rate (assumes one LR group)
        print(f"{model_name}, EPOCH {epoch+1}/{NUM_EPOCHS}")
        train_model(model, train_loader, dataset, OPTIMIZER, CRITERION, SCHEDULER, epoch)
        test_model(model, val_loader, dataset, CRITERION, epoch, type="val")

# Save trained model
#torch.save(model.state_dict(), f'utils/saved_models/{model.name}_config{config.NUM_CONFIG}.pth')

# Test