from typing import List
import pandas as pd
import numpy as np

def token_ranges_to_token_text(text: List[str], tokens: str) -> List[str]:
    """Given a story text and a set of token range, retrieve the corresponding text.
    
    Args:
        text (List[str]): the text in form of list of word
        tokens (str): the token string 
            Ex. 1 -> 20:40 the text is formed by the 20th token to the 40th excluded.
            Ex. 2 -> 544:565,579:582,589:599,608:637 this set of tokens represent a text formed by different part of the text, each portion divided by comma is a part.
    """    
    if text is np.nan: return []
    _answer: List[str] = []
    try:
        _listen_token: List[str] = tokens.split(",")
    except Exception as ex:
        print("Situazione di errore..")
        print("Ricevuto:")
        print(f"Tokens {tokens}")
        return []
    if len(_listen_token) == 0 or "-1" in _listen_token[0]: return []
    
    for token in _listen_token:
        start_idx, end_idx = token.split(":")
        if int(start_idx) > len(text) or int(end_idx) > len(text): return []
        for index in range(int(start_idx)-1, int(end_idx)-1): 
            _answer.append(text[index])
    _answers_without_repetitions = set(_answer)
    return list(_answers_without_repetitions)

if __name__ == "__main__":
    # Load ids files
    test_story_ids = pd.read_csv("./dataset/test_story_ids.csv")
    train_story_ids = pd.read_csv("./dataset/train_story_ids.csv")
    val_story_ids = pd.read_csv("./dataset/dev_story_ids.csv")

    # Load data already tokenized "by using ws as token separator"
    news_qa_data_tokenized = pd.read_csv(
        "./dataset/newsqa-data-tokenized-v1.csv", sep=",")
    news_qa_data_tokenized.question = news_qa_data_tokenized.question.fillna(
        "")
    
    # Drop useless columns
    news_qa_data_tokenized = news_qa_data_tokenized.drop(
        columns=["sentence_starts", "answer_char_ranges"])
    
    # Split data
    news_qa_train = pd.merge(
        train_story_ids, news_qa_data_tokenized, how="inner", on="story_id")
    news_qa_train.is_answer_absent = news_qa_train.is_answer_absent.replace("?", "0.5")
    news_qa_train.is_question_bad = news_qa_train.is_question_bad.replace("?", "0.5")
    news_qa_train = news_qa_train.astype({"is_question_bad": np.float64})

    news_qa_val = pd.merge(
        val_story_ids, news_qa_data_tokenized, how="inner", on="story_id")
    news_qa_val.is_answer_absent = news_qa_val.is_answer_absent.replace("?", "0.5")
    news_qa_val.is_question_bad = news_qa_val.is_question_bad.replace("?", "0.5")
    news_qa_val = news_qa_val.astype({"is_question_bad": np.float64})

    news_qa_test = pd.merge(
        test_story_ids, news_qa_data_tokenized, how="inner", on="story_id")
    news_qa_test.is_answer_absent = news_qa_test.is_answer_absent = news_qa_test.is_answer_absent.replace(
        "?", "0.5")
    news_qa_test.is_question_bad = news_qa_test.is_question_bad = news_qa_test.is_question_bad.replace(
        "?", "0.5")
    news_qa_test = news_qa_test.astype({"is_question_bad": np.float64})
    
    # remove the rows with answers with too many ranges of tokens (> 4)
    news_qa_train["token_ranges_number"] = news_qa_train.answer_token_ranges.str.count(":")
    news_qa_train = news_qa_train[news_qa_train.token_ranges_number <= 4]
    
    news_qa_val["token_ranges_number"] = news_qa_val.answer_token_ranges.str.count(":")
    news_qa_val = news_qa_val[news_qa_val.token_ranges_number <= 4]
    
    news_qa_test["token_ranges_number"] = news_qa_test.answer_token_ranges.str.count(":")
    news_qa_test = news_qa_test[news_qa_test.token_ranges_number <= 4]
    
    # Tokenize the story text
    news_qa_train.story_text = news_qa_test.story_text.str.split(" ")
    
    news_qa_val.story_text = news_qa_val.story_text.str.split(" ")
    
    news_qa_test.story_text = news_qa_test.story_text.str.split(" ")
    
    # # Parse the answer ranges from token to plain text
    news_qa_train['answer'] = news_qa_train.apply(lambda row : token_ranges_to_token_text(row.story_text,row.answer_token_ranges), axis = 1)
    
    news_qa_val['answer'] = news_qa_val.apply(lambda row : token_ranges_to_token_text(row.story_text,row.answer_token_ranges), axis = 1)
    
    news_qa_test['answer'] = news_qa_test.apply(lambda row : token_ranges_to_token_text(row.story_text,row.answer_token_ranges), axis = 1)
    
    print("ok")