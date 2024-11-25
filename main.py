from torchvision import transforms
import pandas as pd

from utils.dataloader import Flickr30kDataset

# GLOBAL VARIABLES
IMAGES_DIR = './data/images/'
CAPTIONS_DIR = './data/captions.csv'


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

# Import dataset
captions_df = pd.read_csv(CAPTIONS_DIR, sep='|', header=None, names=['image_id', 'comment_number', 'caption'], skipinitialspace=True)
dataset = Flickr30kDataset(captions_df, IMAGES_DIR, transform)

# Split in train, validation and test