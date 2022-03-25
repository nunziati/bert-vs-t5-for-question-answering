from tqdm import tqdm
import torch
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader

f1_score = load_metric('f1')

huggingface_model = 't5-base'
batch_size = 32
seed = 7
num_train_epochs = 40
learning_rate = 1e-5
num_workers = 10
class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, truncate_strategy="truncate"):
        self.tokenizer = tokenizer
        self.input = []
        self.target = []

        truncate = truncate_strategy == "truncate"

        for row in tqdm(hf_dataset):
            encoded_input = tokenizer(f"question: {row['question']}  context: {row['plot']}",
                return_tensors="pt", truncation=True, return_overflowing_tokens=True)
            encoded_target = tokenizer(row['answers'][0] if len(row['answers']) > 0 else "",
                return_tensors="pt", truncation=True, return_overflowing_tokens=True)

            if truncate_strategy == "truncate":
                self.input.append(encoded_input.input_ids)
                self.target.append(encoded_target.input_ids)
            else:
                if encoded_input['num_truncated_tokens'] + encoded_target['num_truncated_tokens'] == 0:
                    self.input.append(encoded_input.input_ids)
                    self.target.append(encoded_target.input_ids)
                else:
                    self.input.append(encoded_input['num_truncated_tokens'])
                    self.target.append(encoded_target['num_truncated_tokens'])

        if len(self.input) != len(self.target):
            raise Exception(
                "something wrong while building the dataset: input and target result in different dimensions")

        self.item_count = len(self.input)

    def __len__(self):
        return self.item_count

    def __getitem__(self, index):
        return self.input[index], self.target[index]
        # QUI POTREMMO RITORNARE ANCHE LE attention_mask DELLE DUE STRINGHE TOKENIZZATE, CI SERVE?

    def pack_minibatch(self, data):
        input, target = zip(*data)
        
        input = torch.stack(input, 0)
        target = torch.stack(target, 0)
        
        return input,target
    
def simple_accuracy(preds, labels):
    return (preds == labels).mean().item()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds).item()
    return {
        "accuracy": acc,
        "f1": f1,
    }

def train(model, tokenizer, optimizer, train_set, validation_set, metric):
    # set training mode on the model
    model.train()
    
    # transfer model to cuda
    model.to('cuda')
    
    epoch_train_loss = 0.
    for epoch in range(num_train_epochs):
        epoch_total_example = 0
        for input_ids,target_ids in tqdm(train_set):
            epoch_total_example += input_ids.shape[0]
            optimizer.zero_grad()
            input_ids = input_ids.to('cuda')
            target_ids = target_ids.to('cuda')
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}:\tloss={epoch_train_loss/epoch_total_example:.4f}")
        model.save_pretrained(f'results/{huggingface_model}/model/checkpoint-{epoch}')
        tokenizer.save_pretrained(f'results/{huggingface_model}/tokenizer/checkpoint-{epoch}')
    print("Training Ended")
    print("Evaluate accuracy...")
    model.eval()
    f1, accuracy = 0., 0.
    num_validation_set_examples = 0
    for input_ids, target_ids in tqdm(validation_set):
        model_predictions = model.generate(input_ids)
        metric.add_batch(predictions=model_predictions, references=target_ids)
        score = acc_and_f1(input_ids, target_ids)
        f1 += score['f1']
        accuracy += score['accuracy']
        num_validation_set_examples += input_ids.shape[0]
    final_score = metric.compute()
    f1 = f1 / num_validation_set_examples
    accuracy = accuracy / num_validation_set_examples
    print(f"Sacreblue: {final_score:.4f}, \t F-1: {f1:.4f}, \t Simple Accuracy: {accuracy:.4f}")
    
# Set seed
set_seed(seed)

if __name__ == '__main__':
    
    _data = load_dataset("duorc", "SelfRC")
    
    
    model = T5ForConditionalGeneration.from_pretrained(huggingface_model)
    tokenizer = T5Tokenizer.from_pretrained(huggingface_model)
    # creating the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Define a metric for accuracy
    metric = load_metric('sacrebleu')
    
    my_trainset = DataLoader(Dataset(_data["train"], tokenizer), batch_size=batch_size, num_workers=num_workers, collate_fn=lambda data: Dataset.pack_minibatch(data))
    my_validation = Dataset(_data["validation"], tokenizer) 
    
    train(model = model,
          tokenizer = tokenizer,
          optimizer = optimizer, 
          train_set = my_trainset,
          validation_set = my_validation,
          metric = metric)