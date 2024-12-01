import torch
import csv
import os
import config
import evaluate
from utils.dataset import FoodDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_CSV = "results/results.csv"

def initialize_storage():
    # Ensure the directories exists
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/statistics", exist_ok=True)
    os.makedirs("utils/saved_models", exist_ok=True)
    # Define header of the file
    header = ["config","model","type","epoch","loss","bleu","rouge","meteor"]
    # Check if CSV file exists; if not, create it with the header
    if not os.path.isfile(RESULTS_CSV):
        with open(RESULTS_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

def write_results(model,epoch,type,loss,metrices):
     with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        model_name = model.name if model != "Ensemble" else "Ensemble"
        loss = f"{loss:.4f}" if loss is not None else "-"
        writer.writerow([config.NUM_CONFIG, model_name, type, epoch + 1, loss,
                         metrices["accuracy"], metrices["bleu"], metrices["rouge"], metrices["meteor"]])

def compute_metrices(prediction, true_captions, metrices_dict):

    """
    - Add accuracy or not necessary?
    - max_order of bleu?
    """
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    bleu1 = bleu.compute(predictions=prediction, references=true_captions, max_order=1)
    bleu2 = bleu.compute(predictions=prediction, references=true_captions, max_order=2)

    res_rouge = rouge.compute(predictions=prediction, references=true_captions)

    res_meteor = meteor.compute(predictions=prediction, references=true_captions)

    # Store results in dictionary and return it
    metrices_dict['bleu'] += bleu1
    metrices_dict['rouge'] += res_rouge
    metrices_dict['meteor'] += res_meteor

    return metrices_dict


def train_model(model, train_loader, dataset, optimizer, criterion, epoch, type="train"):
    model.train()
    total_loss = 0
    metrices = {'accuracy':0,
                'bleu':0,
                'rouge':0,
                'meteor':0}

    for images, captions in train_loader:
        print("IMAGES:", images)
        print("CAPTIONS:", captions)
        if isinstance(captions, list):  # Ensure captions is a list of tensors
            # reorginize each tensor (each index across tensors are the captition of an image)
            captions = torch.stack(captions, dim=1)
        print("REORGANIZE CAPTIONS:", captions)
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, captions)
        print("ORIGINAL OUTPUT SHAPE",outputs.shape)
        outputs = outputs.permute(0, 2, 1)  # Now shape is [batch_size, seq_len, vocab_size]
        print("PERMUTED OUTPUT SHAPE",outputs.shape)

        # Loss
        loss = criterion(outputs,captions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        outputs = outputs.argmax(dim=-1) 
        predicted_texts = []
        for sequence in outputs:
            sentence = []
            for idx in sequence:
                idx = int(idx)
                sentence.append(dataset.idx2word[idx])  # Convert index to word
            predicted_texts.append(sentence)  # Join words to form a sentence
        predicted_texts = [dataset.idx2word[idx] for idx in predicted_texts]
        print(predicted_texts)
        true_texts = [dataset.idx2word[idx] for idx in captions.cpu().numpy().flatten()]
        # Metrices
        metrices = compute_metrices(predicted_texts,true_texts,metrices)

    metrices = {key: value / len(train_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)

def test_model(model, test_loader, dataset, gt, criterion, epoch, type="test"):
    model.eval()
    total_loss = 0
    metrices = {'accuracy':0,
                'bleu':0,
                'rouge':0,
                'meteor':0}
    
    with torch.no_grad():
        for images, captions in test_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            # Output
            outputs = model(images)

            # Loss
            loss = criterion(outputs,captions)
            total_loss += loss.item()

            # Metrices
            metrices = compute_metrices(outputs,gt,metrices)

    metrices = {key: value / len(test_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)  
    

