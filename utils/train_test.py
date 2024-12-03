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
    header = ["config","model","type","epoch","loss","accuracy","bleu1","bleu2","rouge","meteor"]
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

    exact_matches = sum([1 if pred == ref[0] else 0 for pred, ref in zip(predictions, references)])
    accuracy = exact_matches / len(predictions) if predictions else 0

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
        print("CAPTIONS:", captions)
        if isinstance(captions, list):  # Ensure captions is a list of tensors
            # reorginize each tensor (each index across tensors are the captition of an image)
            captions = torch.stack(captions, dim=1)
        #print("REORGANIZE CAPTIONS:", captions)
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, captions)
        #print("ORIGINAL OUTPUT SHAPE",outputs.shape)
        outputs = outputs.permute(0, 2, 1)  # Now shape is [batch_size, seq_len, vocab_size]
        #print("PERMUTED OUTPUT SHAPE",outputs.shape)

        # Loss
        loss = criterion(outputs,captions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        outputs = outputs.argmax(dim=-2)

        predicted_texts = []
        for sequence in outputs:
            print("\nSEQUENCE", sequence, "\nSHAPE sequence", sequence.shape)
            sentence = []
            for idx in sequence:
                idx = int(idx)
                sentence.append(dataset.idx2word[idx])  # Convert index to word
            print("SENTENCE", sentence, "\nSHAPE sentence", len(sentence))
            predicted_texts.append([sentence])  # Join words to form a sentence
        if isinstance(predicted_texts[0], list):  # Check if it's a list of lists
            predicted_texts = [idx for sublist in predicted_texts for idx in sublist]
        print("\nKeys in idx2word:", dataset.idx2word.keys())
        print("\nValues in idx2word:", dataset.idx2word.items())
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
    

