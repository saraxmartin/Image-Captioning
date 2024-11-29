import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re
import nltk
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

class Flickr30kDataset(Dataset):
    def __init__(self, captions_df, images_dir, transform=None, type="char"):
        self.captions_df = captions_df
        self.image_dir = images_dir
        self.transform = transform
        # Save data properties for statistics
        self.data_properties = {'image_width': [],
                                'image_height': [],
                                'title_length': [],
                                'lexicon': set()}
        self.all_words = []
        # Save images and its captions
        self.images = []
        self.captions = {}
        processed_images = set()  # set to track already processed images
        
        for image_path in os.listdir(images_dir):
            full_path = os.path.join(images_dir, image_path)
            # Check if file is image
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if image_path in processed_images:  
                continue
            
            # Save image path
            self.images.append(full_path)
            processed_images.add(image_path)
            
            # Get captions for image
            image_captions = captions_df[captions_df["image_name"] == image_path]["caption"].tolist()
            if image_captions:  
                self.captions[full_path] = image_captions
            else:
                continue
            
            # Load image to get properties
            with Image.open(full_path) as img:
                width, height = img.size
                self.data_properties['image_width'].append(width)
                self.data_properties['image_height'].append(height)

            # Calculate statistics from captions
            for caption in image_captions:
                self.data_properties['title_length'].append(len(caption.split()))
                self.data_properties['lexicon'].update(caption.split())
                self.all_words.extend(caption.split())

        # Convert the lexicon set to a list
        self.data_properties['lexicon'] = list(self.data_properties['lexicon'])
        
    def __len__(self):
        return len(self.images)
    
    def preprocess_captions(sent):
        lemmatizer = WordNetLemmatizer()
        sent = sent.lower()
        sent = re.sub("[0-9]","",sent)     # digits
        sent = re.sub("https?://\S+","",sent)     # URLs
        sent = re.sub("@\S+","",sent)     # @'s
        sent = re.sub("[+-/*,':%$#&!_<>(){}^]","",sent)     # special characters
        sent = re.sub(chr(34), "", sent)  # double quote ""  
        sent = re.sub(" +"," ",sent)     # extra spaces
        words = nltk.word_tokenize(sent)
        stop_words = set(nltk.corpus.stopwords.words("english"))
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        sent = ' '.join(words)
        sent = "<start> " + sent + " <end>"
        return sent

    def get_statistics(self, printed):
        plot_statistics(self.data_properties, printed)
        word_statistics(self.all_words, printed)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        captions = self.captions[image_id]
        image = Image.open(image_id).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, captions

def FoodDataset(Dataset):
    def __init__(self, captions_df, images_dir, transform=None, type="char"):
        self.captions_df = captions_df
        self.image_dir = images_dir
        self.transform = transform
        # Save data properties for statistics
        self.data_properties = {'image_width': [],
                                'image_height': [],
                                'title_length': [],
                                'lexicon': set()}
        self.all_words = []
        # Save images and its captions
        self.images = []
        self.captions = []
        
        for image_path in os.listdir(images_dir):
            full_path = os.path.join(images_dir, image_path)
            # Check if file is image
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Obtain image name without ending format
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save image path
            self.images.append(full_path)
            
            # Get captions for image
            image_caption = captions_df[captions_df["Image_Name"] == image_name]["Title"]
            self.captions.append(image_caption)
            
            # Load image to get properties
            with Image.open(full_path) as img:
                width, height = img.size
                self.data_properties['image_width'].append(width)
                self.data_properties['image_height'].append(height)

            # Calculate statistics from captions
            self.data_properties['title_length'].append(len(image_caption.split()))
            self.data_properties['lexicon'].update(image_caption.split())
            self.all_words.extend(image_caption.split())

        # Convert the lexicon set to a list
        self.data_properties['lexicon'] = list(self.data_properties['lexicon'])
        
    def __len__(self):
        return len(self.images)
    
    def preprocess_captions(sent):
        lemmatizer = WordNetLemmatizer()
        sent = sent.lower()
        sent = re.sub("[0-9]","",sent)     # digits
        sent = re.sub("https?://\S+","",sent)     # URLs
        sent = re.sub("@\S+","",sent)     # @'s
        sent = re.sub("[+-/*,':%$#&!_<>(){}^]","",sent)     # special characters
        sent = re.sub(chr(34), "", sent)  # double quote ""  
        sent = re.sub(" +"," ",sent)     # extra spaces
        words = nltk.word_tokenize(sent)
        stop_words = set(nltk.corpus.stopwords.words("english"))
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        sent = ' '.join(words)
        sent = "<start> " + sent + " <end>"
        return sent

    def get_statistics(self, printed):
        plot_statistics(self.data_properties, printed)
        word_statistics(self.all_words, printed)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        caption = self.captions[idx]
        image = Image.open(image_id).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption

def bar_plots(width_stats, height_stats, title_length_stats):
    # Bar plot for Image Width and Height
    plt.figure(figsize=(10, 6))
    width_height_stats = ['count', 'mean', 'min', 'max']
    width_values = [width_stats[key] for key in width_height_stats]
    height_values = [height_stats[key] for key in width_height_stats]

    x = range(len(width_height_stats))  # x-axis positions
    bar1 = plt.bar(x, width_values, width=0.4, label="Width", color=sns.color_palette("Set2")[0], alpha=0.7)
    bar2 = plt.bar([p + 0.4 for p in x], height_values, width=0.4, label="Height", color=sns.color_palette("Set2")[1], alpha=0.7)

    plt.xticks([p + 0.2 for p in x], width_height_stats)  # Adjust x-ticks to center
    plt.title('Image Width and Height Statistics')
    plt.ylabel('Value')
    plt.legend()
    
    # Add numbers on top of bars
    for bar in bar1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
    for bar in bar2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.savefig("results/statistics/width_height_stats.png")

    # Bar plot for Title Length
    plt.figure(figsize=(6, 4))
    title_length_keys = ['mean', 'min', 'max']
    title_length_values = [title_length_stats[key] for key in title_length_keys]

    bar3 = sns.barplot(x=title_length_keys, y=title_length_values, palette="Set2")

    # Add numbers on top of bars
    for i, bar in enumerate(bar3.patches):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", 
                ha='center', va='bottom', fontsize=10)
        
    plt.title('Caption Title Length Statistics')
    plt.ylabel('Value')
    plt.savefig("results/statistics/caption_length_stats.png")

def plot_distributions(image_width, image_height, title_length):
    # Plot Image width and height Distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(image_width, kde=True, palette="Set2", bins=50)
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(image_height, kde=True, palette="Set2", bins=50)
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("results/statistics/image_size.png")

    # Plot Title Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(title_length, kde=True, palette="Set2", bins=50)
    plt.title('Caption Title Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig("results/statistics/caption_lenght.png")


def plot_statistics(data_properties, printed=False):
    # Ensure lengths of image_width and image_height match
    assert len(data_properties['image_width']) == len(data_properties['image_height']), \
        "Image width and height arrays must have the same length."

    # Extract the data
    image_width = data_properties['image_width']
    image_height = data_properties['image_height']
    title_length = data_properties['title_length']

    # 1. Summary statistics
    def calculate_summary(data):
        return {
            "count": len(data),
            "mean": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }
    
    width_stats = calculate_summary(image_width)
    height_stats = calculate_summary(image_height)
    title_length_stats = calculate_summary(title_length)

    # 2. Plot summary statistics
    bar_plots(width_stats, height_stats, title_length_stats)
    
    if printed:
        print("Summary statistics:")
        print("Width:", width_stats)
        print("Height:", height_stats)
        print("Title Length:", title_length_stats)

    # 3. Plot Image Width and Height Distributions
    plot_distributions(image_width, image_height, title_length)


def word_statistics(all_words, printed=False):
    # Calculate word frequencies
    word_freq = Counter(all_words)

    # Common and rare words
    most_common = word_freq.most_common(10)
    least_common = word_freq.most_common()[:-11:-1]

    if printed:
        print("\nWord and character statistics:")
        print("Most common words:")
        for word, freq in most_common:
            print(f"{word}: {freq}")

        print("Least common words:")
        for word, freq in least_common:
            print(f"{word}: {freq}")

    # Visualize word frequencies (Top 10)
    common_words, common_counts = zip(*most_common)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(common_words), y=list(common_counts), palette="Set2")
    plt.title('Top 10 Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.savefig("results/statistics/common_words.png")

    # Visualize characters
    char_freq = Counter("".join(all_words))
    most_common_chars = char_freq.most_common(10)

    chars, char_counts = zip(*most_common_chars)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(chars), y=list(char_counts), palette="Set2")
    plt.title('Top 10 Most Common Characters')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.savefig("results/statistics/common_chars.png")
