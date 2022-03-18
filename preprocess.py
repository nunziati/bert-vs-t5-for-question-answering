import pandas as pd
import numpy as np

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
    