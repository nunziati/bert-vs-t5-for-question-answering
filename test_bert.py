from __future__ import print_function
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import  AdamW, set_seed,  pipeline, BertForQuestionAnswering, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np 

from collections import Counter
import string
import re
import argparse
import json
import sys
import pickle 
from utils.parse_duorc import parse as duorc_parse
from utils.parse_squad import parseForBert as parse_squad
 

batch_size = 16
seed = 7
num_train_epochs = 40
learning_rate = 1e-4
num_workers = 10
max_input_length = 512


def question_answer(model, tokenizer, question, text):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([segment_ids]).to(device),)
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = ""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    
    return answer
    

    
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

    for ground_truths, prediction in zip(gold_answers["input_ids"], predictions["input_ids"]):
        total += 1
        # Remove pad token
        prediction = prediction[prediction!=0]
        ground_truths = ground_truths[ground_truths!=0]
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    return f1, exact_match


if __name__ == '__main__':
    
    _data = load_dataset("duorc","RC")
    
    model = BertForQuestionAnswering.from_pretrained("./results/bert")
    tokenizer = AutoTokenizer.from_pretrained("./results/bert")
    
    test_set = _data["test"]
    
    device = "cuda:0"

    model.to(device)
    
    model.eval()
    
    texts = []
    questions = []
    targets = []
    model_predictions = []
    with torch.no_grad():
        for example in tqdm(test_set):
            for i in range(len(example['answers'])):
                questions.append(example['question'])
                texts.append(example['plot'])
                targets.append(example['answers'][i] if len(example['answers']) > 0 else "")
        for index in tqdm(range(len(questions))):
            model_predictions.append(question_answer(model, tokenizer, questions[index], texts[index]))
        model_predictions = tokenizer(
                                model_predictions,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        targets = tokenizer(
                                targets,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        with open('model_predictions.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(model_predictions, f, pickle.HIGHEST_PROTOCOL)
        with open('targets.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(targets, f, pickle.HIGHEST_PROTOCOL)
    # with open('model_predictions.pickle', 'rb') as f:
    # # The protocol version used is detected automatically, so we do not
    # # have to specify it.
    #     model_predictions = pickle.load(f)
    # with open('targets.pickle', 'rb') as f:
    # # The protocol version used is detected automatically, so we do not
    # # have to specify it.
    #     targets = pickle.load(f)
    f1, em = evaluate(targets, model_predictions)
    print(100*f1/len(model_predictions["input_ids"]),100*em/len(model_predictions["input_ids"]))
        