# system utilities
import re
import string

# tokenization utilitise
from nltk import word_tokenize

# torch
import torch
from torch.utils.data import dataset, dataloader

# import bert
from transformers import AutoTokenizer, BertForTokenClassification

# seed device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TOKENS = {
    "U": 0, # normal word
    "OC": 1, # first capital
    "E.": 3, # period
    "E?": 4, # question mark
    "E!": 5, # exclaimation mark
    "E,": 6, # exclaimation mark
}

# create the dataset loading function
class UtteranceBoundaryDataset(dataset.Dataset):
    raw_data: list[str]
    max_length: int
    tokenizer: AutoTokenizer
    window: int

    # initalization function (to read data, etc.)
    def __init__(self, f, tokenizer, window=10, max_length=100):
        # read the file
        with open(f, 'r') as df:
            d =  df.readlines()
        # store the raw data cleaned
        self.raw_data = [i.strip() for i in d]
        # store window size
        self.window = window
        # store max length
        self.max_length = max_length
        # store tokenizer
        self.tokenizer = tokenizer

    # clean and conform the sentence
    def __call__(self, passage):
        """prepare passage

        Attributes:
            passage (str): the input passage
        """

        # store tokenizer
        tokenizer = self.tokenizer

        # clean sentence
        sentence_raw = re.sub(r' ?(\W)', r'\1', passage)
        # "tokenize" into words
        sentence_tokenized = sentence_raw.split(' ')

        # generate labels by scanning through words
        labels = []
        # iterate through words for labels
        for word in sentence_tokenized:
            # if the first is capitalized
            if word[0].isupper():
                labels.append(TOKENS["OC"])
            # otherwise, if the last is a punctuation, append
            elif word[-1] in ['.', '?', '!', ',']:
                labels.append(TOKENS[f"E{word[-1]}"])
            # finally, if nothing works, its just regular
            else:
                labels.append(TOKENS["U"])

        # remove symbols and lower
        sentence_tokenized = [re.sub(r'[.?!,]', r'', i) for i in sentence_tokenized]

        # tokenize time!
        tokenized = tokenizer(sentence_tokenized,
                              truncation=True,
                              is_split_into_words=True,
                              max_length=self.max_length)

        # and now, we get result
        # not sure why, but if there is multiple items for each
        # of the tokens, we only calculate error on one and leave
        # the rest as -100

        final_labels = []
        prev_word_idx = None

        # for each tokens
        for elem in tokenized.word_ids(0):
            # if its none, append nothing
            if elem is None:
                final_labels.append(-100)
            # if its not none, append something
            # if its a new index
            elif elem != prev_word_idx:
                # find the label
                final_labels.append(labels[elem])
                # set prev
                prev_word_idx = elem
            # otherwise, append skip again
            else:
                final_labels.append(-100)

        # set labels
        tokenized["labels"] = final_labels

        return tokenized

    # get a certain item
    def __getitem__(self, index):
        # get the raw data shifted by sentence
        sents = self.raw_data[index*self.window:index*self.window+self.window]
        # prepare the sentence and return
        return self(" ".join(sents))

    def __len__(self):
        return len(self.raw_data)//self.window

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data = UtteranceBoundaryDataset("./data/Pitt.txt", tokenizer)
data[10]

data("hello, I am one of those people. What's happening here?")
tokenizer.convert_ids_to_tokens(data("hello, I am one of those people. What's happening here?")["input_ids"])

