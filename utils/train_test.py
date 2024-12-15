import torch
import csv
import os
import evaluate
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import sys

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

def train_model(model, train_loader, dataset, optimizer, criterion, scheduler, epoch, type="train"):
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
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, captions)
        #print("Shape of outputs before reshape:", outputs.shape)

        # Get predictions
        _, preds = torch.max(outputs, dim=2)
        #print("PREDS",preds)

        # Reshape output to [batch_size * sequence_lenght, vocab_size]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        #print("Shape of outputs after reshape:", outputs.shape)

        # Flatten captions to [batch_size * seq_len]
        #print("Shape of captions:", captions.shape)
        captions_flat = captions.reshape(-1)
        #print("Reshape of captions:", captions_flat.shape)

        # Compute the loss
        loss = criterion(outputs, captions_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Get the predicted captions
        predicted_texts, true_texts = get_captions(preds,captions,dataset)
        
        #print("\n\nPREDICTED TEXTS: ", predicted_texts)
        #print("\nTRUE TEXTS: ", true_texts)

        # Metrices
        metrices = compute_metrices(predicted_texts,true_texts,metrices)

    metrices = {key: value / len(train_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)
    write_captions_results(model,type,epoch,predicted_texts,true_texts)

    scheduler.step()

def test_model(model, test_loader, dataset, criterion, epoch, type="test"):
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

            # Output
            outputs = model(images,captions)

            # Predictions
            _, preds = torch.max(outputs,dim=2)

            # Loss
            outputs = outputs.reshape(-1, outputs.shape[-1])
            captions_flat = captions.reshape(-1)
            loss = criterion(outputs,captions_flat)
            total_loss += loss.item()

            # Get the predicted captions
            predicted_texts, true_texts = get_captions(preds,captions,dataset)

            # Metrices
            metrices = compute_metrices(predicted_texts,true_texts,metrices)

    metrices = {key: value / len(test_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)
    write_captions_results(model,type,epoch,predicted_texts,true_texts)
    
    # Saving metrics and doing plots 
    #csv_file_path = "./results/results.csv"
    #plot_metrics_and_save(csv_file_path, metrics=['Bleu1', 'Bleu2', 'Rouge', 'Meteor1'])
    
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



