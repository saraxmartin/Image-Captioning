import torch
import csv
import os
import evaluate
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage.transform import resize
import math
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

# Import the config module
import config

# Load evaluate data
meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_CSV = "results/results.csv"

def initialize_storage():
    # Ensure the directories exists
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/statistics", exist_ok=True)
    os.makedirs("utils/saved_models", exist_ok=True)
    results_csv = 'results/results.csv'
    captions_csv = f'results/captions_{config.NUM_CONFIG}.csv'
    header1 = ['Config', 'Model', 'Type', 'Epoch', 'Loss', 'Accuracy', 'Bleu1' , 'Bleu2', 'Rouge', 'Meteor']
    header2 = ['Config', 'Model', 'Type', 'Epoch', 'Predicted', 'GT']
    # Check if CSV file exists; if not, create it with the header
    #if not os.path.isfile(results_csv):
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header1)
    #if not os.path.isfile(captions_csv):
    with open(captions_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header2)

def write_results(model,epoch,type,loss,metrices):
     with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        model_name = model.name if model != "Ensemble" else "Ensemble"
        #model_name = model
        loss = f"{loss:.4f}" if loss is not None else "-"
        writer.writerow([config.NUM_CONFIG, model_name, type, epoch + 1, loss,
                         metrices["accuracy"], metrices["bleu1"], metrices["bleu2"], metrices["rouge"], metrices["meteor"]])

def write_captions_results(model, type, epoch, predicted_texts, true_texts):
    # Storing the image with its corresponding prediction and gt captions
    with open(f'results/captions_{config.NUM_CONFIG}.csv', mode='a', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        # Iterate over your images and captions
        for predicted_caption, ground_truth_caption in zip(predicted_texts, true_texts):
            # Write a row for each image
            csv_writer.writerow([config.NUM_CONFIG, model.name, type, epoch+1, predicted_caption, ground_truth_caption])
            #csv_writer.writerow([config.NUM_CONFIG, model, type, epoch+1, predicted_caption, ground_truth_caption])

def clean_lists(pred, gt):
    cleaned_pred = []
    cleaned_gt = []
    for p, g in zip(pred, gt):
        # Remove <SOS>, <EOS>, and <PAD> from gt
        g_cleaned = [word for word in g if word not in ('<SOS>', '<EOS>', '<PAD>')]
        # Find indices to retain in pred
        retain_indices = [i for i, word in enumerate(g) if word not in ('<SOS>', '<EOS>', '<PAD>')]
        # Filter pred based on retain indices
        p_cleaned = [p[i] for i in retain_indices]
        
        # Append cleaned lists
        cleaned_gt.append(g_cleaned)
        cleaned_pred.append(p_cleaned)
    
    return cleaned_pred, cleaned_gt

def compute_metrices(prediction, true_captions, metrices_dict):
    # Clean captions
    pred_cleaned, gt_cleaned = clean_lists(prediction, true_captions)
    pred_cleaned = [' '.join(p) for p in pred_cleaned]
    gt_cleaned = [[' '.join(g)] for g in gt_cleaned]

    # Compute blue, rouge and meteor metrices
    bleu1 = bleu.compute(predictions=pred_cleaned, references=gt_cleaned, max_order=1)['bleu'] # ['bleu'] --> to get actual score
    bleu2 = bleu.compute(predictions=pred_cleaned, references=gt_cleaned, max_order=2)['bleu']
    res_rouge = rouge.compute(predictions=pred_cleaned, references=gt_cleaned)
    rouge_L = res_rouge['rougeL']  # Extract ROUGE-L 
    res_meteor = meteor.compute(predictions=pred_cleaned, references=gt_cleaned)["meteor"]
    
    # Compute accuracy
    total = 0
    count = 0
    for pred, ref in zip(pred_cleaned, gt_cleaned):
        """for p, r in zip(pred,ref[0]):
            print(f"P {p}, Ref: {r}")
            if p == r:
                count+=1
            total+=1"""
        # Compare token by token
        # Split the predictions and references into words (tokens)
        pred_tokens = pred.split()  # Split predicted sentence into tokens
        ref_tokens = ref[0].split()  # Split reference sentence into tokens

        # Compare each word in the prediction with the reference
        for p, r in zip(pred_tokens, ref_tokens):
            total += 1
            #print(f"P {p}, Ref: {r}")
            if p == r:
                count += 1
            
        #print(f"PREDICTION: {pred}, REFERENCE: {ref}")
    
    accuracy = count/ total if pred_cleaned else 0

    # Store results in dictionary and return it
    metrices_dict['accuracy'] = metrices_dict.get('accuracy', 0) + accuracy
    metrices_dict['bleu1'] = metrices_dict.get('bleu1', 0) + bleu1
    metrices_dict['bleu2'] = metrices_dict.get('bleu2', 0) + bleu2
    metrices_dict['rouge'] = metrices_dict.get('rouge', 0) + rouge_L
    metrices_dict['meteor'] = metrices_dict.get('meteor', 0) + res_meteor

    return metrices_dict

def get_captions(preds,gt,dataset):
    # Get predicted captions
    pred_captions = []
    for i in range(preds.shape[0]):
        caption = []
        for j in range(preds.shape[1]):
            word_idx = preds[i,j].item()
            word = dataset.idx2word[word_idx]
            #if word == "<EOS>":
                #break
            caption.append(word)
        pred_captions.append(caption)
    # Get the ground truth captions
    true_captions = []
    for caption in gt:
        caption_words = [dataset.idx2word[word_idx.item()] for word_idx in caption]
        true_captions.append(caption_words)
    return pred_captions, true_captions

def convert_captions(gt,dataset):
    # Get the ground truth captions
    true_captions = []
    for caption in gt:
        caption_words = [dataset.idx2word[word_idx.item()] for word_idx in caption]
        true_captions.append(caption_words)
    return true_captions

def generate_valid_filename(caption_list):
    #remove special tokens <SOS>, <EOS>, and <PAD>
    filtered_caption = [word for word in caption_list if word not in ['<SOS>', '<EOS>', '<PAD>']]
    
    #join words into a string
    caption_string = ' '.join(filtered_caption)
    
    #replace spaces and other invalid characters with underscores
    valid_filename = caption_string.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    #optionally, limit the length of the filename to avoid too long filenames
    valid_filename = valid_filename[:255]  #limit to 255 characters (common max length for filenames)
    
    return valid_filename

def plot_attention(image, attention_weights, grid_size=None, type="train", caption="", save_dir=None):
    """
    Plot the original image and its attention map side by side, and save the result to a directory.
    
    Args:
        image (torch.Tensor): The original image tensor (C, H, W).
        attention_weights (torch.Tensor): Attention weights (num_regions, seq_len) or (num_regions,).
        grid_size (tuple): Optional, Grid size of attention weights (e.g., (7, 7)).
        type (str): Type of the dataset (train, val, test) to display in the title.
        caption (str): Caption to be displayed on the original image and used in the saved filename.
        save_dir (str): Directory to save the attention maps. If None, saves to the default location.
    """
    # Ensure attention_weights is 2D (num_regions, seq_len)
    if attention_weights.dim() == 1:  # If attention_weights is (num_regions,)
        attention_weights = attention_weights.unsqueeze(-1)  # Reshape to (num_regions, 1)

    num_regions = attention_weights.size(0)  # Number of regions
    seq_len = attention_weights.size(1)  # Sequence length (number of words)

    # Infer grid size dynamically if not provided
    if grid_size is None:
        side_length = int(math.sqrt(num_regions))
        grid_size = (side_length, side_length)

    # Correct save directory path
    if save_dir is None:
        # Default save path: results/attention_maps in the root of the project
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'attention_maps')
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot attention for each timestep
    for t in range(seq_len):
        attention_map = attention_weights[:, t].reshape(grid_size).detach().cpu().numpy()
        attention_resized = resize(attention_map, (image.shape[1], image.shape[2]), mode='reflect')

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        ax[0].imshow(image.permute(1, 2, 0).cpu())
        ax[0].set_title(f"Original Image ({type.capitalize()})")
        ax[0].axis('off')

        # Add the caption to the image (overlay text)
        ax[0].text(0.5, 0.05, caption, ha='center', va='bottom', color='white', fontsize=12, weight='bold', transform=ax[0].transAxes)
        
        # Plot attention overlay
        ax[1].imshow(image.permute(1, 2, 0).cpu())
        ax[1].imshow(attention_resized, cmap='jet', alpha=0.5)
        ax[1].set_title(f"Attention Map ({type.capitalize()} - Word {t+1})")
        ax[1].axis('off')

        # Save the figure with the caption in the filename
        # Replace spaces with underscores to ensure a valid filename
        save_path = os.path.join(save_dir, f"{caption}_attention_{t+1}.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to avoid display

def train_model(model, train_loader, dataset, optimizer, criterion, scheduler, epoch, VOCAB_SIZE, type="train"):
    model.train()
    total_loss = 0
    metrices = {'accuracy':0,
                'bleu1':0,
                'bleu2':0,
                'rouge':0,
                'meteor':0}

    for images, captions in train_loader:
        #print("IMAGES:", images)
        #print("CAPTIONS:", captions)
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        tar = captions
        optimizer.zero_grad()
        
        true_captions = convert_captions(captions, dataset)

        # Forward pass
        outputs, att_weights = model(images, captions)
        outputs_new = outputs
        #print("Outputs shape:", outputs.shape)
        #print("Shape of outputs before reshape:", outputs.shape)
        outputs = outputs.view(-1, VOCAB_SIZE)
        target = tar.contiguous().view(-1)
        #print("Outputs shape:", outputs.shape)

        # Get predictions
        _, preds = torch.max(outputs_new, dim=2)
        #print("PREDS",preds)

        # Compute the loss
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Get the predicted captions
        predicted_texts, true_texts = get_captions(preds,captions,dataset)
        
        #print("\n\nPREDICTED TEXTS: ", predicted_texts)
        #print("\nTRUE TEXTS: ", true_texts)

        # Metrices
        metrices = compute_metrices(predicted_texts,true_texts,metrices)
        last_images = images.detach().cpu()[-3:]
        last_attention_weights = att_weights.detach().cpu()[-3:]
        last_captions = true_captions[-3:]

    metrices = {key: value / len(train_loader) for key, value in metrices.items()}
    loss = total_loss/ len(train_loader)
    write_results(model,epoch,type,loss,metrices)
    write_captions_results(model,type,epoch,predicted_texts,true_texts)

    scheduler.step()

    for i in range(len(last_images)):
        current_caption = generate_valid_filename(last_captions[i])
        plot_attention(last_images[i], last_attention_weights[i],caption=current_caption,type=type)

def test_model(model, test_loader, dataset, criterion, epoch, VOCAB_SIZE, type="test"):
    model.eval()
    total_loss = 0
    metrices = {'accuracy':0,
                'bleu1':0,
                'bleu2':0,
                'rouge':0,
                'meteor':0}
    
    with torch.no_grad():
        for images, captions in test_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            tar = captions
            true_captions = convert_captions(captions, dataset)

            # Forward pass
            outputs, att_weights = model(images, captions)
            outputs_new = outputs
            #print("Outputs shape:", outputs.shape)
            #print("Shape of outputs before reshape:", outputs.shape)
            outputs = outputs.view(-1, VOCAB_SIZE)
            target = tar.contiguous().view(-1)
            #print("Outputs shape:", outputs.shape)

            # Get predictions
            _, preds = torch.max(outputs_new, dim=2)
            #print("PREDS",preds)

            # Loss
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # Get the predicted captions
            predicted_texts, true_texts = get_captions(preds,captions,dataset)

            # Metrices
            metrices = compute_metrices(predicted_texts,true_texts,metrices)
            last_images = images.detach().cpu()[-3:]
            last_attention_weights = att_weights.detach().cpu()[-3:]
            last_captions = true_captions[-3:]

            if att_weights  is not None:
                for i in range(min(len(images), 3)):  # Show attention maps for up to 3 images
                    current_caption = generate_valid_filename(last_captions[i])
                    plot_attention(last_images[i], last_attention_weights[i],caption= current_caption,type='val')


    metrices = {key: value / len(test_loader) for key, value in metrices.items()}
    loss = total_loss/ len(test_loader)
    write_results(model,epoch,type,loss,metrices)
    write_captions_results(model,type,epoch,predicted_texts,true_texts)

    
def plot_metrics_and_save(csv_file_path, output_folder=None, metrics=['Loss', 'Accuracy']):
    """
    Plots the specified metrics for train and test sets for different models and saves the plots as images.

    Parameters:
    - csv_file_path (str): Full path to the CSV file.
    - output_folder (str): Path to the folder where images will be saved (default is the same directory as the CSV file).
    - metrics (list): List of metrics to plot (default is ['Loss', 'Accuracy']).
    """
    #load the data
    data = pd.read_csv(csv_file_path)

    #ensure the 'Type' column is treated as categorical, not numerical
    data['Type'] = data['Type'].astype('category')

    #if no output folder is specified, use the directory of the CSV file
    if output_folder is None:
        output_folder = os.path.dirname(csv_file_path)

    #make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    #loop through each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model in data['Model'].unique():
            for data_type in data['Type'].cat.categories:
                subset = data[(data['Model'] == model) & (data['Type'] == data_type)]
                if not subset.empty:
                    plt.plot(subset['Epoch'], subset[metric], label=f'{model} - {data_type}')

        plt.title(f'{metric} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()

        #save the plot
        sanitized_metric = metric.replace(' ', '_').lower()
        file_name = f"{sanitized_metric}_plot.png"
        save_path = os.path.join(output_folder, file_name)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()  # Close the plot to free memory



