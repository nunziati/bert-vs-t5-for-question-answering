from __future__ import print_function
from typing import List, Tuple
from tqdm import tqdm
import torch
import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import Counter
import argparse

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description='CLI for training T5 T2T model')
    
    parser.add_argument('--t5_model', type=str, default="t5-base",
                        help="What type of T5 model do you want use?")    
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')
    
    parser.add_argument('--workers', type=int, default=10,
                        help='number of working units used to load the data (default: 10)')
    
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')
    
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')
    
    parsed_arguments = parser.parse_args()

    return parsed_arguments
class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset, tokenizer: transformers.models.t5.tokenization_t5.T5Tokenizer):
        """Constructor for Dataset class
        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): HuggingFace Dataset
            tokenizer (transformers.models.t5.tokenization_t5.T5Tokenizer): HuggingFace Tokenizer

        Raises:
            Exception: if questions and answers have different length it will raise an exception
        """        
        self.tokenizer = tokenizer
        self.inputs_text: List[str] = []
        self.targets_text: List[str] = []
        
        for row in tqdm(hf_dataset):
            self.inputs_text.append(f"question: {row['question']}  context: {row['plot']}")
            self.targets_text.append(row['answers'][0] if len(row['answers']) > 0 else "")
            
        if len(self.inputs_text) != len(self.targets_text):
            raise Exception(
                "something wrong while building the dataset: input and target result in different dimensions")

        self.item_count: int = len(self.inputs_text)

    def __len__(self):
        """Magic method over-ride for class lenght evaluation

        Returns:
            int: lenght of the object 
        """
        return self.item_count

    def __getitem__(self, index: int):
        """Magic method over-ride for class getitem method

        Args:
            index (int): index for identify question-context and answer example

        Returns:
            Tuple(str,str): (Question-Context, Answer)
        """
        return self.inputs_text[index], self.targets_text[index]

    @staticmethod
    def pack_minibatch(data: List[Tuple[str,str]], max_input_length: int):
        """Pack mini-batch function

        Args:
            data (List[Tuple]): Inputs for the model and the associated target
            max_input_length (int): Max allowed lenght for the model

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: (Input tokenized and encode, Attention mask, Target tokenized and encoded)
        """
        inputs, targets = zip(*data)
        encoded_inputs: transformers.tokenization_utils_base.BatchEncoding = tokenizer(
                                inputs,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        encoded_targets: transformers.tokenization_utils_base.BatchEncoding = tokenizer(
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
    """_summary_

    Args:
        prediction (_type_): _description_
        ground_truth (_type_): _description_

    Returns:
        _type_: _description_
    """
    if ground_truth.shape[0] == prediction.shape[0]: 
        if (ground_truth == prediction).all():
            return 1
    return 0

def f1_score(prediction, ground_truth):
    """_summary_

    Args:
        prediction (_type_): _description_
        ground_truth (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        predictions (_type_): _description_
        gold_answers (_type_): _description_

    Returns:
        _type_: _description_
    """
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        # Remove pad token
        prediction = prediction[prediction!=0]
        ground_truths = ground_truths[ground_truths!=-100]
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    return f1, exact_match

def train(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizer, optimizer: AdamW, train_set: DataLoader, validation_set: DataLoader, num_train_epochs: int, device: str, batch_size: int):
    """_summary_

    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (DataLoader): _description_
        validation_set (DataLoader): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    # set training mode on the model
    model.train()
    
    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        epoch_total_example = 0
        for input_ids,maskeds_attention,target_ids in tqdm(train_set):
            epoch_total_example += input_ids.shape[0]
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            maskeds_attention = maskeds_attention.to(device)
            outputs = model(input_ids=input_ids, attention_mask=maskeds_attention, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/epoch_total_example:.4f}")
        model.eval()
        _f1 = _exact_match = 0
        with torch.no_grad():
            for input_ids, maskeds_attention, target_ids in tqdm(validation_set):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                maskeds_attention = maskeds_attention.to(device)
                model_predictions = model.generate(input_ids=input_ids, attention_mask=maskeds_attention)
                # F1 over each batch
                f1, exact_match = evaluate(model_predictions, target_ids)
                _f1 += f1_score
                _exact_match += exact_match
        f1 = 100.0 * f1 / len(validation_set)
        exact_match = 100.0 * exact_match / len(validation_set)
        
        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            model.save_pretrained(f'results/{model.name_or_path}/model/best-f1')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 5 == 0:
            model.save_pretrained(f'results/{model.name_or_path}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}')
        model.train()
        
    model.save_pretrained(f'results/{model.name_or_path}/model/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}')
        


if __name__ == '__main__':
    args = parse_command_line_arguments()
    
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))
    
    # Set seed
    set_seed(args.seed)

    _data = load_dataset("duorc", "SelfRC")
    
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    # creating the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    _train_set = Dataset(_data["train"], tokenizer)
    _validation_set = Dataset(_data["validation"], tokenizer)
    my_trainset_dataloader = DataLoader(_train_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: Dataset.pack_minibatch(data, args.max_input_length))
    my_validation_dataloader = DataLoader(_validation_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: Dataset.pack_minibatch(data, args.max_input_length))
    
    train(model = model,
          tokenizer = tokenizer,
          optimizer = optimizer, 
          train_set = my_trainset_dataloader,
          validation_set = my_validation_dataloader,
          num_train_epochs=args.epochs,device=args.device, batch_size=args.batch_size)