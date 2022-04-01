from collections import Counter
from typing import List, Tuple
import datasets
import transformers
import torch
from tqdm import tqdm


class DatasetMap():
    @staticmethod
    def duorc(example):
        nr_answer = len(example["answers"])
        return [example["plot"]]*nr_answer, [example["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]]

    @staticmethod
    def squad(example):
        nr_answer = len(example["answers"]["text"])
        return [example["context"]]*nr_answer, [example["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]["text"]]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset, tokenizer, parser=None):
        """Constructor for Dataset class
        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): HuggingFace Dataset
            tokenizer: HuggingFace Tokenizer

        Raises:
            Exception: if two between questions, answers and contexts have different length it will raise an exception
        """
        self.tokenizer = tokenizer
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.contexts: List[str] = []

        for row in tqdm(hf_dataset):
            _contexts, _questions, _answers = parser(row)

            self.contexts += _contexts
            self.questions += _questions
            self.answers += _answers

        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.contexts):
            raise Exception(
                "something wrong while building the dataset: questions, contexts and answers result in different dimensions")

        self.item_count: int = len(self.questions)

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
            Tuple(str,str,str): (Context, Question, Answer)
        """
        return self.contexts[index], self.questions[index], self.answers[index]

    def pack_minibatch(self, data: List[Tuple[str, str]]):
        """Pack mini-batch function

        Args:
            data (Tuple[List[str],List[str],List[str]]): (Contexts, Questions, Answers)

        Returns:
            Tuple[List[str],List[str],List[str]]: (Contexts, Questions, Answers)
        """
        return zip(*data)

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
                return 1
        return 0

    def __f1_score(self, prediction_tokens, ground_truth_tokens):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate(self, predictions, gold_answers):
        """_summary_

        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_

        Returns:
            _type_: _description_
        """
        f1 = exact_match = 0

        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # Remove pad token
            tokens_to_remove = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            }
            prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
            ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
            f1 += self.__f1_score(prediction, ground_truths)
            exact_match += self.__exact_match_score(prediction, ground_truths)
        return 100*f1/len(predictions), 100*exact_match/len(predictions)
