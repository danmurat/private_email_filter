import json
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Sets up our given dataset to be compatable with training classifier ML models
"""


class PreProcess:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.v_train_text = None
        self.v_test_text = None
        self.word_counts_threshold_indexed = None

    # (X_train, y_train, X_test, y_test)
    # to be given to DataObj after preprocessing
    def get_data(self) -> tuple:
        # if one is set up, they'll all be setup, so pointless checking
        is_data_setup = (
            self.v_train_text is not None
        )  #!= None and self.train_data.any() != None and self.v_test_text.any() != None and self.test_data.any() != None

        if is_data_setup:
            return (
                self.v_train_text,
                self.train_data["label"],
                self.v_test_text,
                self.test_data["label"],
            )
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def get_train_test_text(self) -> tuple[pd.Series, pd.Series]:
        train, test = self._load_dataset()

        return train["text"], test["text"]

    # yep pretty much 50/50 (vary slight variation)
    def is_data_imbalanced(self) -> None:
        train, test = self._load_dataset()

        print("length of train data = ", len(train))
        print("length of ham = ", len(train[train["label"] == 0]))

    def get_training_data(self) -> pd.DataFrame:
        if self.train_data is not None:
            return self.train_data
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def get_testing_data(self) -> pd.DataFrame:
        if self.test_data is not None:
            return self.test_data
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def get_vectorised_train_text(self) -> NDArray[np.float64]:
        if self.v_train_text is not None:
            return self.v_train_text
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def get_vectorised_test_text(self) -> NDArray[np.float64]:
        if self.v_test_text is not None:
            return self.v_test_text
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def get_word_counts_thresehold_indexed(self) -> dict[str, int]:
        if self.word_counts_threshold_indexed is not None:
            return self.word_counts_threshold_indexed
        else:
            raise ValueError(
                "Data missing (type None). Run preprocess() first or check data integrity."
            )

    def preprocess_tfidf(self) -> None:
        train, test = self._load_dataset()
        self.train_data = train
        self.test_data = test
        self.fit_tfidf(train["text"], test["text"])

    # after testing against our own BoW, this fares better. Including so we can save
    # text data in this format for encrypted training and testing.
    def fit_tfidf(self, train_text: pd.Series, test_text: pd.Series) -> None:
        tfidf_vec = TfidfVectorizer(
            stop_words="english", max_features=3020, min_df=2, max_df=0.8
        )

        tfidf_train = tfidf_vec.fit_transform(train_text)
        tfidf_test = tfidf_vec.transform(test_text)

        self.v_train_text = tfidf_train
        self.v_test_text = tfidf_test

    # RUN THIS SO VARIABLES GET ASSIGNED!
    def preprocess_bow(self) -> None:
        print("loading datasets...")

        data = self._load_dataset()
        self.train_data = self._transform_text(data[0])
        self.test_data = self._transform_text(data[1])

        print("setting up word count dictionaries...")

        # TODO: we should probably be accessing ALL text, not just train
        word_counts = self._get_word_counts(self.train_data["text"])
        word_counts_threshold = self._word_counts_thresehold(word_counts, 200)
        self.word_counts_threshold_indexed = self._word_counts_thresehold_indexed(
            word_counts_threshold
        )
        # into json so we can access it later for preprocessing single emails (in demo)
        self._save_indexed_words_100(self.word_counts_threshold_indexed)

        print("tokenising text...")

        tokenised_train_text = self.tokenise_text(
            self.train_data["text"], self.word_counts_threshold_indexed
        )
        tokenised_test_text = self.tokenise_text(
            self.test_data["text"], self.word_counts_threshold_indexed
        )

        print("vectorising dataset text for training...")

        dimensions = len(self.word_counts_threshold_indexed)
        self.v_train_text = self.vectorise_text(tokenised_train_text, dimensions)
        self.v_test_text = self.vectorise_text(tokenised_test_text, dimensions)

        print("\npreprocessing complete!\n")

    # for demo purposes
    def preprocess_single_email(
        self, mail_df: pd.DataFrame, indexedDict: dict[str, int]
    ) -> NDArray[np.float64]:

        # print("transforming text...")
        transformed_data = self._transform_text(mail_df)

        # word counts already set up (and we saved into a seperate file), so we can skip this

        # print("tokenising text...")
        tokenised = self.tokenise_text(transformed_data["text"], indexedDict)

        # print("vectorising text...")
        vector = self.vectorise_text(tokenised, len(indexedDict))

        # print("complete!\n")

        return vector

    def vectorise_text(
        self, tokenised_text_list: list[list[int]], dimensions: int
    ) -> NDArray[np.float64]:
        entries = len(tokenised_text_list)
        # v will hold all vectorised text (will have v[0] = 0, v[1] = 1, etc..)
        v = np.zeros((entries, dimensions))

        # finds indexes for each email and sets those positions in v to 1 (since they occur)
        for e in range(entries):
            for i in tokenised_text_list[e]:
                v[e, i] = 1
        return v

    # this transforms the og text string into a list of numbers (which are the top100 indexed)
    def tokenise_text(
        self, text_data: pd.Series, word_counts_threshold_indexed: dict[str, int]
    ) -> list[list[int]]:
        # loop through Text
        tokenised_text_list: list[list[int]] = []
        # top100_indexed = self._top100Indexed()
        for t in text_data:
            # splits to list of words (rather than single string of sentences)
            t_splitwords: list[str] = t.split()
            token_list: list[int] = []
            # check if each word is in dict and transform to index
            for i in range(len(t_splitwords)):
                if t_splitwords[i] in word_counts_threshold_indexed:
                    token_list.append(word_counts_threshold_indexed[t_splitwords[i]])

            tokenised_text_list.append(token_list)

        return tokenised_text_list

    # each word is given a number (index) instead of frequency now, so we can vectorise
    # each email for the model (fixed dimensions)
    def _word_counts_thresehold_indexed(
        self, word_counts_threshold: dict[str, int]
    ) -> dict[str, int]:
        word_counts_threshold_indexed = {}
        # wordCounts100 = self._wordCounts100()
        for i, key in enumerate(word_counts_threshold):
            word_counts_threshold_indexed[key] = i

        return word_counts_threshold_indexed

    # we'll keep the words that have been used at least n times. (default 100)
    def _word_counts_thresehold(
        self, word_counts: dict[str, int], threshold: int = 100
    ) -> dict[str, int]:
        # word_counts = self._getWordCounts(self.train_data["text"])
        word_counts_threshold = {}
        for key in word_counts:
            if word_counts[key] >= threshold:
                word_counts_threshold[key] = word_counts[key]

        return word_counts_threshold

    # builds a frequency count for each word using dictionaries (hashmap)
    def _get_word_counts(self, text: pd.Series) -> dict[str, int]:
        word_counts = {}

        for textrow in text:
            words_list = textrow.split()
            for word in words_list:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1  # simply creates

        return word_counts

    def _transform_text(self, data: pd.DataFrame) -> pd.DataFrame:
        non_alphanumeric = re.compile(
            r"(\n|\W)"
        )  # we include \n too to get out the way
        numbers = re.compile(r"(\d+)")
        whitespaces = re.compile(r"(\s+)")
        rejoin_s = re.compile(
            r"(\s{1}s\s{1})"
        )  # any s with single space before and after
        common = re.compile(r"(\sa\s|\sand\s|\sthe\s|\sthat\s|\sof\s|\swith\s|\sas\s)")

        data["text"] = data["text"].str.replace(non_alphanumeric, " ", regex=True)

        data["text"] = data["text"].str.replace(numbers, "numbers", regex=True)

        data["text"] = data["text"].str.replace(whitespaces, " ", regex=True)

        data["text"] = data["text"].str.replace(rejoin_s, "s ", regex=True)

        data["text"] = data["text"].str.replace(common, "", regex=True)

        return data

    def _load_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train: pd.DataFrame = pd.read_json("../spam_dataset/train.jsonl", lines=True)
        test: pd.DataFrame = pd.read_json("../spam_dataset/test.jsonl", lines=True)

        return train, test

    def _save_indexed_words_100(self, dict: dict[str, int]) -> None:
        with open("indexed100.json", "w") as file:
            json.dump(dict, file, indent=4)
