import torch
import csv
import os
import config
import evaluate
import torch.nn as nn


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_CSV = "results/results.csv"

def initialize_storage():
    # Ensure the directories exists
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/statistics", exist_ok=True)
    os.makedirs("utils/saved_models", exist_ok=True)
    

def write_results(model,epoch,type,loss,metrices):
     with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        model_name = model.name if model != "Ensemble" else "Ensemble"
        loss = f"{loss:.4f}" if loss is not None else "-"
        writer.writerow([config.NUM_CONFIG, model_name, type, epoch + 1, loss,
                         metrices["accuracy"], metrices["bleu1"], metrices["bleu2"], metrices["rouge"], metrices["meteor"]])

def clean_sentence(tokens, remove_tokens={"<SOS>", "<EOS>", "<PAD>"}):
    return [token for token in tokens if token not in remove_tokens]

def compute_metrices(prediction, true_captions, metrices_dict):

    """
    - Add accuracy or not necessary?
    - max_order of bleu?
    """
    predictions = [" ".join(clean_sentence(pred)) for pred in prediction]
    references = [[" ".join(clean_sentence(reference))] for reference in true_captions]
    
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    bleu1 = bleu.compute(predictions=predictions, references=references, max_order=1)['bleu'] # ['bleu'] --> to get actual score
    bleu2 = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu']

    res_rouge = rouge.compute(predictions=predictions, references=references)
    rouge_L = res_rouge['rougeL']  # Extract ROUGE-L 

    res_meteor = meteor.compute(predictions=predictions, references=references)["meteor"]
    total = 0
    count = 0
    for pred, ref in zip(predictions, references):
        for p, r in zip(pred,ref[0]):
            if p == r:
                count+=1
            total+=1
            
        print(f"Prediction: {pred}, Reference: {ref}")
    
    accuracy = count/ total if predictions else 0

    # Store results in dictionary and return it
    metrices_dict['accuracy'] = metrices_dict.get('accuracy', 0) + accuracy
    metrices_dict['bleu1'] = metrices_dict.get('bleu1', 0) + bleu1
    metrices_dict['bleu2'] = metrices_dict.get('bleu2', 0) + bleu2
    metrices_dict['rouge'] = metrices_dict.get('rouge', 0) + rouge_L
    metrices_dict['meteor'] = metrices_dict.get('meteor', 0) + res_meteor

    return metrices_dict

def train_model(model, train_loader, dataset, optimizer, criterion, epoch, type="train"):
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
        #print("REORGANIZE CAPTIONS:", captions)
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, captions)

        # Permute the output to shape [batch_size, vocab_size, seq_len]
        outputs = outputs.permute(0, 2, 1)  # Now shape is [batch_size, seq_len, vocab_size]
        #print("Shape of outputs after permute:", outputs.shape)

        # Flatten the outputs to [batch_size * seq_len, vocab_size]
        outputs_flat = outputs.contiguous().view(-1, outputs.size(2))  # Flatten output (batch_size * seq_len, vocab_size)
        #print("Shape of outputs_flat:", outputs_flat.shape)

        # Flatten captions to [batch_size * seq_len]
        captions_flat = captions.view(-1)  # Flatten captions (batch_size * seq_len)
        #print("Shape of captions_flat:", captions_flat.shape)

        # Create mask for PAD (0), SOS (1), and EOS (2)
        mask = (captions_flat != 0) & (captions_flat != 1) & (captions_flat != 2)  # Mask PAD (0), SOS (1), EOS (2)

        #print("Shape of mask:", mask.shape)
        # Reshape outputs_flat to [batch_size * seq_len, vocab_size]
        outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, outputs.size(1))  # [batch_size * seq_len, vocab_size]

        # Apply mask to both the outputs_flat and captions_flat
        outputs_flat = outputs_flat[mask]  # Apply mask to output
        captions_flat = captions_flat[mask]  # Apply mask to captions


        #print("Shape of outputs_flat after mask:", outputs_flat.shape)
        #print("Shape of captions_flat after mask:", captions_flat.shape)

        # Now you can calculate the loss
        loss = criterion(outputs_flat, captions_flat)

        # Loss
        pad_idx = 0
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Use pad_idx for exclusion in CrossEntropyLoss
        loss = criterion(outputs_flat, captions_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        outputs = outputs.argmax(dim=-2)

        predicted_texts = []
        for sequence in outputs:
            #print("\nSEQUENCE", sequence, "\nSHAPE sequence", sequence.shape)
            sentence = []
            for idx in sequence:
                idx = int(idx)
                sentence.append(dataset.idx2word[idx])  # Convert index to word
            #print("SENTENCE", sentence, "\nSHAPE sentence", len(sentence))
            predicted_texts.append([sentence])  # Join words to form a sentence
        if isinstance(predicted_texts[0], list):  # Check if it's a list of lists
            predicted_texts = [idx for sublist in predicted_texts for idx in sublist]
        #print("\nKeys in idx2word:", dataset.idx2word.keys())
        #print("\nValues in idx2word:", dataset.idx2word.items())
        predicted_texts = [dataset.idx2word[int(idx)] if str(idx).isdigit() else idx for idx in predicted_texts]

        true_texts = []
        for sentence in captions.cpu().numpy():
            true_texts.append([dataset.idx2word[idx] for idx in sentence])


        #print("\n\nPREDICTED TEXTS: ", predicted_texts)
        #print("\nTRUE TEXTS: ", true_texts)

        # Metrices
        metrices = compute_metrices(predicted_texts,true_texts,metrices)

    metrices = {key: value / len(train_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)

def test_model(model, test_loader, dataset, criterion, epoch, type="test"):
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
            metrices = compute_metrices(outputs,captions,metrices)

    metrices = {key: value / len(test_loader) for key, value in metrices.items()}
    write_results(model,epoch,type,total_loss,metrices)  
    

