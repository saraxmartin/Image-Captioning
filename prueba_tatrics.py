import torch
import csv
import os
import evaluate
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import sys

meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
def compute_metrices(prediction, true_captions):
    # Clean captions
    pred_cleaned, gt_cleaned = prediction, true_captions
    pred_cleaned = [' '.join(p) for p in pred_cleaned]
    gt_cleaned = [[' '.join(g)] for g in gt_cleaned]
    print(pred_cleaned)
    print(gt_cleaned)

    # Compute blue, rouge and meteor metrices
    bleu1 = bleu.compute(predictions=pred_cleaned, references=gt_cleaned, max_order=1)['bleu'] # ['bleu'] --> to get actual score
    print("BLEU1", bleu1)
    bleu2 = bleu.compute(predictions=pred_cleaned, references=gt_cleaned, max_order=2)['bleu']
    print("BLEU2", bleu2)
    res_rouge = rouge.compute(predictions=pred_cleaned, references=gt_cleaned)
    rouge_L = res_rouge['rougeL']  # Extract ROUGE-L 
    print("ROUGE_L", rouge_L)
    res_meteor = meteor.compute(predictions=pred_cleaned, references=gt_cleaned)["meteor"]
    print("RES_METEOR", res_meteor)
    
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
    print("AC: ", accuracy)

metrices = compute_metrices(["Me", "llamo", "lara"],["Me", "reí", "ayer"])
