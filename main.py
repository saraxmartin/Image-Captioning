from torchvision import transforms
import pandas as pd

from utils.dataloader import Flickr30kDataset
from utils.train_test import initialize_storage, train_model, test_model

# GLOBAL VARIABLES
IMAGES_DIR = './data/images/'
CAPTIONS_DIR = './data/captions.csv'
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 0.8, 0.1, 0.1

initialize_storage()

# Data transfpormation
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

# Import dataset
captions_df = pd.read_csv(CAPTIONS_DIR, sep='|', header=None, names=['image_id', 'comment_number', 'caption'], skipinitialspace=True)
dataset = Flickr30kDataset(captions_df, IMAGES_DIR, transform)

# Split in train, validation and test


# Train and validation loop



# Test