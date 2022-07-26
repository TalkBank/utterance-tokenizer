# system utilities
import re
import string
import random

# tokenization utilitise
from nltk import word_tokenize

# torch
import torch
from torch.utils.data import dataset 
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

# import huggingface utils
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification

# tqdm
from tqdm import tqdm

# wandb
import wandb

# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TOKENS = {
    "U": 0, # normal word
    "OC": 1, # first capital
    "E.": 2, # period
    "E?": 3, # question mark
    "E!": 4, # exclaimation mark
    "E,": 5, # exclaimation mark
}

# weights and biases
hyperparametre_defaults = dict(
    learning_rate = 3.5e-5,
    batch_size = 15,
    epochs = 4,
    window = 10
)

# start wandb
run = wandb.init(project='jemoka', entity='utok', config=hyperparametre_defaults, mode="disabled")
# run = wandb.init(project='jemoka', entity='utok', config=hyperparametre_defaults)

# set configuration
config = run.config

# create the dataset loading function
class UtteranceBoundaryDataset(dataset.Dataset):
    raw_data: list[str]
    max_length: int
    tokenizer: AutoTokenizer
    window: int

    # initalization function (to read data, etc.)
    # max length doesn't matter
    def __init__(self, f, tokenizer, window=10, max_length=1000):
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

# create the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load the data (train using MICASE, test on Pitt)
train_data = UtteranceBoundaryDataset("./data/AphasiaBankEnglishProtocol.txt", tokenizer, window=config.window)
test_data = UtteranceBoundaryDataset("./data/Pitt.txt", tokenizer, window=config.window)

# create data collator utility on the tokenizer
data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

# load the data
train_dataloader = iter(DataLoader(train_data,
                                   batch_size=config.batch_size,
                                   shuffle=True, collate_fn=lambda x:x))
test_dataloader = iter(DataLoader(test_data,
                                  batch_size=config.batch_size,
                                  shuffle=True, collate_fn=lambda x:x))

# create the model and tokenizer
model = BertForTokenClassification.from_pretrained("bert-base-uncased",
                                                   num_labels=len(TOKENS)).to(DEVICE)
optim = AdamW(model.parameters(), lr=config.learning_rate)

# utility to move a whole dictionary to a device
def move_dict(d, device):
    """move a dictionary to device

    Attributes:
        d (dict): dictionary to move
        device (torch.Device): device to move to
    """

    for key, value in d.items():
        d[key] = d[key].to(device)

# start training!
val_data = list(iter(test_dataloader))

# for each epoch
for epoch in range(config.epochs):
    print(f"Training epoch {epoch}")

    # for each batch
    for indx, batch in tqdm(enumerate(iter(train_dataloader)), total=len(train_dataloader)):
        # pad and conform batch
        batch = data_collator(batch)
        move_dict(batch, DEVICE)

        # train!
        output = model(**batch)
        # backprop
        output.loss.backward()
        # step
        optim.step()
        optim.zero_grad()

        # log!
        run.log({
            'loss': output.loss.cpu().item()
        })

        # if need to validate, validate
        if indx % 10 == 0:
            # select a val batch
            val_batch = data_collator(random.choice(val_data))
            move_dict(val_batch, DEVICE)
            # run!
            output = model(**val_batch)
            # log!
            run.log({
                'val_loss': output.loss.cpu().item()
            })

# write model down
model.save_pretrained(f"./models/{run.name}")
tokenizer.save_pretrained(f"./models/{run.name}")

