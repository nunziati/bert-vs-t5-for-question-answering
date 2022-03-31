from collections import Counter
from typing import List,Tuple
import datasets
import transformers
import torch 
from tqdm import tqdm

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

    def pack_minibatch(self,data: List[Tuple[str,str]], max_input_length: int):
        """Pack mini-batch function

        Args:
            data (List[Tuple]): Inputs for the model and the associated target
            max_input_length (int): Max allowed lenght for the model

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: (Input tokenized and encode, Attention mask, Target tokenized and encoded)
        """
        inputs, targets = zip(*data)
        encoded_inputs: transformers.tokenization_utils_base.BatchEncoding = self.tokenizer(
                                inputs,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        encoded_targets: transformers.tokenization_utils_base.BatchEncoding = self.tokenizer(
                                targets,
                                padding="longest",
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt",
                            )
        
        input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        encoded_targets = encoded_targets.input_ids
        
        # replace padding token id's of the labels by -100, crossEntropy skip target label == -100
        encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100
        
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
    f1 = exact_match = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        # Remove pad token
        prediction = prediction[prediction!=0]
        ground_truths = ground_truths[ground_truths!=-100]
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    return f1, exact_match