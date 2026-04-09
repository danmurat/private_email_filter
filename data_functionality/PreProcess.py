import pandas as pd
import numpy as np
import re
import json

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
    def getData(self):
        # if one is set up, they'll all be setup, so pointless checking
        is_data_setup = self.v_train_text.any() #!= None and self.train_data.any() != None and self.v_test_text.any() != None and self.test_data.any() != None
        
        if is_data_setup:
            return (self.v_train_text, self.train_data["label"], self.v_test_text, self.test_data["label"])
        else:
            raise ValueError("Data missing (type None). Run preprocess() first or check data integrity.")
   

    def getTrainingData(self):
        return self.train_data
    
    def getTestingData(self):
        return self.test_data

    def getVectorisedTrainText(self):
        return self.v_train_text

    def getVectorisedTestText(self):
        return self.v_test_text
    
    def getWordCountsThresholdIndexed(self):
        return self.word_counts_threshold_indexed

    # RUN THIS SO VARIABLES GET ASSIGNED!
    def preprocess(self):
        print("loading datasets...")

        data = self._loadDataset()
        self.train_data = self._transformText(data[0])
        self.test_data = self._transformText(data[1])

        print("setting up word count dictionaries...")
        
        # TODO: we should probably be accessing ALL text, not just train
        word_counts = self._getWordCounts(self.train_data["text"])
        word_counts_threshold = self._wordCountsThreshold(word_counts, 100)
        self.word_counts_threshold_indexed = self._wordCountsThresholdIndexed(word_counts_threshold)
        # into json so we can access it later for preprocessing single emails (in demo)
        self._saveIndexedWords100(self.word_counts_threshold_indexed)
        
        print("tokenising text...")
        
        tokenised_train_text = self.tokeniseText(self.train_data["text"], self.word_counts_threshold_indexed)
        tokenised_test_text = self.tokeniseText(self.test_data["text"], self.word_counts_threshold_indexed)

        print("vectorising dataset text for training...")

        dimensions = len(self.word_counts_threshold_indexed)
        self.v_train_text = self.vectoriseText(tokenised_train_text, dimensions)
        self.v_test_text = self.vectoriseText(tokenised_test_text, dimensions)

        print("\npreprocessing complete!\n")

    # for demo purposes
    def preprocessSingleEmail(self, mail_df, indexedDict):

        #print("transforming text...")
        transformed_data = self._transformText(mail_df)

        # word counts already set up (and we saved into a seperate file), so we can skip this

        #print("tokenising text...")
        tokenised = self.tokeniseText(transformed_data["text"], indexedDict)

        #print("vectorising text...")
        vector = self.vectoriseText(tokenised, len(indexedDict))

        #print("complete!\n")

        return vector



    def vectoriseText(self, tokenised_text_list, dimensions):
        entries = len(tokenised_text_list)
        # v will hold all vectorised text (will have v[0] = 0, v[1] = 1, etc..)
        v = np.zeros((entries, dimensions))

        # finds indexes for each email and sets those positions in v to 1 (since they occur)
        for e in range(entries):
            for i in tokenised_text_list[e]:
                v[e, i] = 1
        return v

    # this transforms the og text string into a list of numbers (which are the top100 indexed)
    def tokeniseText(self, text_data, word_counts_threshold_indexed) -> list:
        #loop through Text
        tokenised_text_list = []
        #top100_indexed = self._top100Indexed()
        for t in text_data:
            # splits to list of words (rather than single string of sentences)
            t_splitwords = t.split()
            token_list = []
            # check if each word is in dict and transform to index
            for i in range(len(t_splitwords)):
                if t_splitwords[i] in word_counts_threshold_indexed:
                    token_list.append(word_counts_threshold_indexed[t_splitwords[i]])
            
            tokenised_text_list.append(token_list)
        
        return tokenised_text_list

    # each word is given a number (index), so we can vectorise each email for the model (fixed dimensions)
    def _wordCountsThresholdIndexed(self, word_counts_threshold) -> dict:
        word_counts_threshold_indexed = {}
        #wordCounts100 = self._wordCounts100()
        for i, key in enumerate(word_counts_threshold):
            word_counts_threshold_indexed[key] = i

        return word_counts_threshold_indexed

    # we'll keep the words that have been used at least n times. (default 100)
    def _wordCountsThreshold(self, word_counts: dict, threshold: int = 100) -> dict:
        # word_counts = self._getWordCounts(self.train_data["text"])
        word_counts_threshold = {}
        for key in word_counts:
            if word_counts[key] >= threshold:
                word_counts_threshold[key] = word_counts[key]

        return word_counts_threshold

    # builds a frequency count for each word using dictionaries (hashmap)
    def _getWordCounts(self, text) -> dict:
        word_counts = {}
        
        for textrow in text:
            words_list = textrow.split()
            for word in words_list:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1 # simply creates
        
        return word_counts 
        
    
    def _transformText(self, data) -> tuple:
        non_alphanumeric = re.compile(r'(\n|\W)') # we include \n too to get out the way
        numbers = re.compile(r'(\d+)')
        whitespaces = re.compile(r'(\s+)')
        rejoin_s = re.compile(r'(\s{1}s\s{1})') # any s with single space before and after
        common = re.compile(r'(\sa\s|\sand\s|\sthe\s|\sthat\s|\sof\s|\swith\s|\sas\s)')

        data["text"] = data["text"].str.replace(non_alphanumeric, " ", regex=True)

        data["text"] = data["text"].str.replace(numbers, "numbers", regex=True)

        data["text"] = data["text"].str.replace(whitespaces, " ", regex=True)

        data["text"] = data["text"].str.replace(rejoin_s, "s ", regex=True)

        data["text"] = data["text"].str.replace(common, "", regex=True)

        return data


    def _loadDataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_json("../spam_dataset/train.jsonl", lines=True)
        test = pd.read_json("../spam_dataset/test.jsonl", lines=True)

        return (train, test)


    

    def _saveIndexedWords100(self, dict):
        with open("indexed100.json", "w") as file:
            json.dump(dict, file, indent=4)