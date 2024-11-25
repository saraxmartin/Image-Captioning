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
from collections import Counter

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
        # Save images and its captions
        self.images = []
        self.captions = {}

        for image_path in os.listdir(images_dir):
            full_path = os.path.join(images_dir, image_path)
            # Check if file is image
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Save image path
            self.images.append(full_path)
            # Get captions for image
            image_captions = captions_df[captions_df["image_name"] == image_path]["comment"].tolist()
            self.captions[image_path] = image_captions

            # Load image to get properties
            with Image.open(full_path) as img:
                width, height = img.size
                self.data_properties['image_width'].append(width)
                self.data_properties['image_height'].append(height)

            # Calculate statistics from captions
            for caption in image_captions:
                self.data_properties['title_length'].append(len(caption.split()))
                self.data_properties['lexicon'].update(caption.split())

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
        sent = re.sub(" +"," ",sent)     # extra spaces
        words = nltk.word_tokenize(sent)
        stop_words = set(nltk.corpus.stopwords.words("english"))
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        sent = ' '.join(words)
        sent = "<start> " + sent + " <end>"
        return sent

    def get_statistics(self):
        return self.data_properties

    def __getitem__(self, idx):
        image_id = self.images.iloc[idx]
        captions = self.captions[image_id]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, captions
    

def plot_statistics(data_properties):
    # Convert data properties to pandas DataFrame for convenience
        data_df = pd.DataFrame({
            'image_width': data_properties['image_width'],
            'image_height': data_properties['image_height'],
            'title_length': data_properties['title_length']
        })

        # 1. Summary statistics for dimensions and title length
        summary = data_df.describe()
        print("Summary statistics:\n", summary)

        # 2. Plot Image Width and Height Distributions
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data_df['image_width'], kde=True, color="blue", bins=20)
        plt.title('Image Width Distribution')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        sns.histplot(data_df['image_height'], kde=True, color="green", bins=20)
        plt.title('Image Height Distribution')
        plt.xlabel('Height (pixels)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # 3. Plot Title Length Distribution
        plt.figure(figsize=(6, 6))
        sns.histplot(data_df['title_length'], kde=True, color="purple", bins=20)
        plt.title('Caption Title Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.show()

        # 4. Lexicon Analysis
        # Calculate word frequencies
        lexicon = data_properties['lexicon']
        word_freq = Counter(lexicon)

        # Common and rare words
        most_common = word_freq.most_common(10)
        least_common = word_freq.most_common()[:-11:-1]

        print("\nMost common words:")
        for word, freq in most_common:
            print(f"{word}: {freq}")

        print("\nLeast common words:")
        for word, freq in least_common:
            print(f"{word}: {freq}")

        # Visualize word frequencies (Top 10)
        common_words, common_counts = zip(*most_common)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(common_words), y=list(common_counts), palette="Blues_r")
        plt.title('Top 10 Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

        # Visualize characters
        char_freq = Counter("".join(lexicon))
        most_common_chars = char_freq.most_common(10)

        chars, char_counts = zip(*most_common_chars)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(chars), y=list(char_counts), palette="Greens_r")
        plt.title('Top 10 Most Common Characters')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.show()