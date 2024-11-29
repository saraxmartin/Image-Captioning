import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import pandas as pd

import config
from utils.dataset import Flickr30kDataset
from utils.train_test import initialize_storage, train_model, test_model
from utils.model import CaptioningModel

# GLOBAL VARIABLES
IMAGES_DIR = 'data/images/'
CAPTIONS_DIR = "data/captions.txt"
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
captions_df = pd.read_csv(CAPTIONS_DIR, sep='|', skiprows=1, header=None, names=['image_name', 'caption_number', 'caption'])
dataset = Flickr30kDataset(captions_df, IMAGES_DIR, transform)
VOCAB_SIZE = len(dataset.data_properties["lexicon"])
#statistics = dataset.get_statistics(printed=True)
print(f"Number of images in total: {len(dataset)}")

# Split in train, validation and test
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
print(f"Number of images in train dataset: {len(train_dataset)}")
print(f"Number of images in validation dataset: {len(val_dataset)}")
print(f"Number of images in test dataset: {len(test_dataset)}")

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Configure models
models = [models.densenet201, models.vgg16, models.resnet50]
names = ["densenet201","vgg16","resnet50"]


for images, captions in train_loader:
    print(len(images))
    print(len(captions))
    print(len(captions[0]))
    print(captions)

# Train and validation loop
for model_function, model_name in zip(models, names):
    model = CaptioningModel(model_function, embed_size, hidden_size, VOCAB_SIZE, attention_size)
    for epoch in range(NUM_EPOCHS):
        train_model(model, train_loader, OPTIMIZER, CRITERION, epoch)
        test_model(model, val_loader, OPTIMIZER, CRITERION, epoch, type="val")

# Save trained model
#torch.save(model.state_dict(), f'utils/saved_models/{model.name}_config{config.NUM_CONFIG}.pth')

# Test
