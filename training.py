from tqdm import tqdm
import torch
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader



huggingface_model = 't5-base'
batch_size = 16
seed = 7
num_train_epochs = 40
learning_rate = 1e-4
num_workers = 10
max_input_length = 512

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, truncate_strategy="truncate"):
        self.tokenizer = tokenizer
        self.inputs_text = []
        self.targets_text = []
        
        for row in tqdm(hf_dataset):
            self.inputs_text.append(f"question: {row['question']}  context: {row['plot']}")
            self.targets_text.append(row['answers'][0] if len(row['answers']) > 0 else "")
        #     encoded_input = tokenizer(f"question: {row['question']}  context: {row['plot']}",
        #         return_tensors="pt", truncation=True, return_overflowing_tokens=True)
        #     encoded_target = tokenizer(row['answers'][0] if len(row['answers']) > 0 else "",
        #         return_tensors="pt", truncation=True, return_overflowing_tokens=True)

        #     if truncate_strategy == "truncate":
        #         self.input.append(encoded_input.input_ids)
        #         self.target.append(encoded_target.input_ids)
        #     else:
        #         if encoded_input['num_truncated_tokens'] + encoded_target['num_truncated_tokens'] == 0:
        #             self.input.append(encoded_input.input_ids)
        #             self.target.append(encoded_target.input_ids)
        #         else:
        #             self.input.append(encoded_input['num_truncated_tokens'])
        #             self.target.append(encoded_target['num_truncated_tokens'])

        if len(self.inputs_text) != len(self.targets_text):
            raise Exception(
                "something wrong while building the dataset: input and target result in different dimensions")

        self.item_count = len(self.inputs_text)

    def __len__(self):
        return self.item_count

    def __getitem__(self, index):
        return self.inputs_text[index], self.targets_text[index]
        # QUI POTREMMO RITORNARE ANCHE LE attention_mask DELLE DUE STRINGHE TOKENIZZATE, CI SERVE?

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
    




def train(model, tokenizer, optimizer, train_set, validation_set, metric):
    f1_score = load_metric('f1')
    accuracy = load_metric('accuracy')
    # set training mode on the model
    model.train()
    
    # transfer model to cuda
    model.to('cuda')
    
    
    
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        epoch_total_example = 0
        for input_ids,maskeds_attention,target_ids in tqdm(train_set):
            epoch_total_example += input_ids.shape[0]
            optimizer.zero_grad()
            input_ids = input_ids.to('cuda')
            target_ids = target_ids.to('cuda')
            maskeds_attention = maskeds_attention.to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=maskeds_attention, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}:\tloss={epoch_train_loss/epoch_total_example:.4f}")
        model.save_pretrained(f'results/{huggingface_model}/model/checkpoint-{epoch}')
        tokenizer.save_pretrained(f'results/{huggingface_model}/tokenizer/checkpoint-{epoch}')
    print("Training Ended")
    # print("Evaluate accuracy...")
    # model.eval()
    # num_validation_set_examples = 0
    # with torch.no_grad():
    #     for input_ids, maskeds_attention, target_ids in tqdm(validation_set):
    #         input_ids = input_ids.to('cuda')
    #         target_ids = target_ids.to('cuda')
    #         maskeds_attention = maskeds_attention.to('cuda')
    #         model_predictions = model.generate(input_ids=input_ids, attention_mask=maskeds_attention,do_sample=False)
    #         metric.add_batch(predictions=model_predictions, references=target_ids)
    #         #accuracy.add_batch(predictions=model_predictions, references=target_ids)
    #         #f1_score.add_batch(predictions=model_predictions, references=target_ids)
    #         num_validation_set_examples += input_ids.shape[0]
    # final_score = metric.compute()
    # #f1 = f1_score.compute()
    # #accuracy = accuracy.compute()
    # print(f"Sacreblue: {final_score:.4f}")
    
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
    
    _train_set = Dataset(_data["train"], tokenizer)
    _validation_set = Dataset(_data["validation"], tokenizer)
    my_trainset_dataloader = DataLoader(_train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda data: Dataset.pack_minibatch(data))
    my_validation_dataloader = DataLoader(_validation_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda data: Dataset.pack_minibatch(data))
    
    train(model = model,
          tokenizer = tokenizer,
          optimizer = optimizer, 
          train_set = my_trainset_dataloader,
          validation_set = my_validation_dataloader,
          metric = metric)