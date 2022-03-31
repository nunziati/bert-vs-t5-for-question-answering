from __future__ import print_function
from tqdm import tqdm
import torch
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed,  pipeline, BertForQuestionAnswering, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np 

from collections import Counter
import string
import re
import argparse
import json
import sys

from utils.parse_duorc import parse as duorc_parse
 
huggingface_model = 't5-base'
batch_size = 16
seed = 7
num_train_epochs = 40
learning_rate = 1e-4
num_workers = 10
max_input_length = 512

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, remove_more_than_1_answer = True, parser = None):
        self.tokenizer = tokenizer
        self.inputs_text = []
        self.targets_text = []
        
        for row in tqdm(hf_dataset):
            _inputs, _targets = parser(row, remove_more_than_1_answer)
            self.inputs_text = self.inputs_text + _inputs
            self.targets_text = self.targets_text + _targets
            
        if len(self.inputs_text) != len(self.targets_text):
            raise Exception(
                "something wrong while building the dataset: input and target result in different dimensions")

        self.item_count = len(self.inputs_text)

    def __len__(self):
        return self.item_count

    def __getitem__(self, index):
        return self.inputs_text[index], self.targets_text[index]
    
    @staticmethod
    def pack_minibatch(data):
        inputs, targets = zip(*data)
        encoded_inputs = tokenizer(
                                inputs,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        encoded_targets = tokenizer(
                                targets,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        
        input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        encoded_targets = encoded_targets.input_ids
        
        # replace padding token id's of the labels by -100, crossEntropy skip target label == -100
        encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100
        
        return input_ids, attention_mask, encoded_targets
    
    
def exact_match_score(prediction, ground_truth):
    if ground_truth.shape[0] == prediction.shape[0]: 
        if (ground_truth == prediction).all():
            return 1
    return 0


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.tolist()
    
    ground_truth_tokens = ground_truth.tolist()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(predictions,gold_answers ):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        # Remove pad token
        prediction = prediction[prediction!=0]
        ground_truths = ground_truths[ground_truths!=-100]
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    return f1, exact_match



if __name__ == '__main__':
    
    _data = load_dataset("duorc","ParaphraseRC")
    
    
    
    model = T5ForConditionalGeneration.from_pretrained("./results/t5-base/model/checkpoint-31")
    tokenizer = T5Tokenizer.from_pretrained("./results/t5-base/tokenizer/checkpoint-31")
    

    _test_set = Dataset(_data["test"], tokenizer, remove_more_than_1_answer=False, parser=duorc_parse)
    my_testset_dataloader = DataLoader(_test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda data: Dataset.pack_minibatch(data))
    
    device = "cuda:0"
    model.to(device)

    model.eval()
    num_validation_set_batch = f1 = exact_match = 0
    with torch.no_grad():
        for input_ids, maskeds_attention, target_ids in tqdm(my_testset_dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            maskeds_attention = maskeds_attention.to(device)
            model_predictions = model.generate(input_ids=input_ids, attention_mask=maskeds_attention)
            # F1 over each batch
            _f1, _exact_match = evaluate(model_predictions, target_ids)
            f1 += _f1
            exact_match += _exact_match
    f1 = 100.0 * f1 / len(_test_set)
    exact_match = 100.0 * exact_match / len(_test_set)
    print(f"\t F1 = {f1:.2f}, EM = {exact_match:.2f}")