from __future__ import print_function
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import BertForQuestionAnswering, AutoTokenizer
import argparse
from MyDataset import Dataset, DatasetMap
 
dataset_instruction = {
    "duorc": {
        "parser": DatasetMap.duorc,
        "test_set": "test"
    },
    "squad": {
        "parser": DatasetMap.squad,
        "test_set": "validation"
    }
}

max_input_length = 512

def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default="./results/bert", help="the name of the model/checkpoint to be used for the classifier (e.g. ./results/checkpoint")
    parser.add_argument('--dataset', default="squad", choices=["squad", "duorc"], help="the name of the dataset to be used for testing")
    parser.add_argument('--subversion', default="", choices=["", "SelfRC", "ParaphraseRC"], help="the name of the subversion of the dataset, in case 'duorc' dataset is selected")
    parser.add_argument('--device', default="cpu", choices=["cpu", "cuda:0", "cuda:1"], help="device selected for performing the evaluation")
    
    parsed_arguments = parser.parse_args()

    return parsed_arguments

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

if __name__ == '__main__':
    # getting the arguments from command line (or default ones)
    args = parse_command_line_arguments()

    dataset_name = args.dataset
    dataset_sub = args.subversion
    model_name = args.model
    device = torch.device(args.device)

    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _data = load_dataset(dataset_name, dataset_sub if dataset_name=="duorc" else None)
    test_set = Dataset(_data[dataset_instruction[dataset_name]["test_set"]], tokenizer, parser=dataset_instruction[dataset_name]["parser"])
    
    model.eval()
    
    texts = []
    questions = []
    targets = []
    model_predictions = []
    with torch.no_grad():
        for context, question, answer in tqdm(test_set):
            questions.append(question)
            texts.append(context)
            targets.append(answer)

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

    f1, em = test_set.evaluate(model_predictions.input_ids.tolist(), targets.input_ids.tolist())
    print(f1, em)