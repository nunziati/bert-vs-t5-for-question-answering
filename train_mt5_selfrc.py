from __future__ import print_function
from typing import List, Tuple
from tqdm import tqdm
import torch

from datasets import load_dataset
from transformers import PreTrainedTokenizer, MT5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
import argparse

from MyDataset import Dataset
import MyDataset


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training MT5 T2T model')

    parser.add_argument('--t5_model', type=str, default="google/mt5-small",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs (default: 40)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=10,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def train(model: MT5ForConditionalGeneration, tokenizer: PreTrainedTokenizer, optimizer: AdamW, train_set: Dataset, validation_set: Dataset, num_train_epochs: int, device: str, batch_size: int, max_input_length: int = 512):
    """_summary_

    Args:
        model (MT5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (Dataset): _description_
        validation_set (Dataset): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_trainset_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                        num_workers=args.workers, collate_fn=lambda data: train_set.pack_minibatch(data))
    my_validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size,
                                          num_workers=args.workers, collate_fn=lambda data: validation_set.pack_minibatch(data))

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for contexts,questions,answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
            encoded_inputs = tokenizer(
                                    inputs,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
            encoded_targets = tokenizer(
                                    answers,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )

            input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for contexts, questions, answers in tqdm(my_validation_dataloader):
                inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                    questions, contexts)))
                encoded_inputs = tokenizer(
                    inputs,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = tokenizer(
                    answers,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                encoded_targets = encoded_targets.input_ids

                encoded_inputs = encoded_inputs.to(device)
                encoded_targets = encoded_targets.to(device)
                attention_mask = attention_mask.to(device)
                model_predictions = model.generate(
                    input_ids=encoded_inputs, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()
        f1, exact_match = validation_set.evaluate(
            target_encoded, model_predictions_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            model.save_pretrained(f'results/{model.name_or_path}/best-f1')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0:
            model.save_pretrained(f'results/{model.name_or_path}/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/tcheckpoint-{epoch+1}')
        model.train()

    model.save_pretrained(
        f'results/{model.name_or_path}/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(
        f'results/{model.name_or_path}/checkpoint-{epoch+1}')


if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)

    _data = load_dataset("duorc", "ParaphraseRC")

    model = MT5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_set = Dataset(_data["train"], tokenizer,
                        parser=MyDataset.DatasetMap.duorc)
    validation_set = Dataset(
        _data["validation"], tokenizer, parser=MyDataset.DatasetMap.duorc)

    train(model=model,
          tokenizer=tokenizer,
          optimizer=optimizer,
          train_set=train_set,
          validation_set=validation_set,
          num_train_epochs=args.epochs, device=args.device, batch_size=args.batch_size)
