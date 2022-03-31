from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import argparse

def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default="bert-base-uncased", choices=["bert-base-uncased", "distilbert-base-uncased", "bert-base-cased", "distilbert-base-uncased", "bert-large-uncased", "bert-large-cased"], help="the name of the model to be used for the classifier")
    parser.add_argument('--out_dir', default="./results", help="the output directory where checkpoints will be saved")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="regularization parameter (default: 0.1)")
    parser.add_argument("--train_batch_size", type=int, default=16, help="mini-batch size during training (default: 16)")
    parser.add_argument("--val_batch_size", type=int, default=16, help="mini-batch size during validation (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument('--eval_strategy', default="steps", choices=["epoch", "steps"], help="when to evaluate the state of the training procedure")
    parser.add_argument('--save_stragety', default="epoch", choices=["epoch", "steps"], help="when to save the checkpoint files during training")
    parser.add_argument('--steps', type=int, default=450, help="how often to save/eval during the training procedure, if the save/eval mode is 'step' (default: every 450 batches)")
    parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")

    parsed_arguments = parser.parse_args()

    return parsed_arguments

def preprocess_function(examples):
    # retriveing the questions
    questions = [q.strip() for q in examples["question"]]

    # tokenization of question and context text
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")

    # retriveing the answers as ranges of tokens in the context
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    # handling of the longest inputs, that have to be truncated
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

if __name__ == '__main__':
    # getting the arguments from command line (or default ones)
    args = parse_command_line_arguments()

    # selecting the model and dataset to be used
    model_name = args.model
    dataset_name = "squad"

    # loading the dataset, model and tokenizer
    squad = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # preprocess the data
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

    # defining the training arguments to be passed to the parameter
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        evaluation_strategy=args.eval_strategy,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy=args.save_stragety,
        log_level="info",
        logging_steps=args.steps,
    )

    # defining the data collator, useful to build the mini-batches
    data_collator = DefaultDataCollator()

    # defining the trainer, specifying the arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # running the training procedure
    trainer.train()