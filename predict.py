# system utilities
import re
import string
import random

# tokenization utilitise
from nltk import word_tokenize, sent_tokenize

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

# seed model
MODEL = "./models/playful-lake-2" 


# seed tokenizers and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = BertForTokenClassification.from_pretrained(MODEL).to(DEVICE)

# eval mode
model.eval()

# get sentence input
sentence = "okay well cat decides to climb up a tree and gets on a branch that it can't come down from and then the little girl comes along and the dog comes along and they both see the ms can they get their father and the father climbs up the tree to try and get the cat you got two people stuck in a tree or two creatures stuck in the tree and the bird is watching all this and somebody calls the fire department and they come out with a ladder and they're they're going to get ready they're gonna they're gonna rescue these two end of story"

# pass it through the tokenizer and model
tokd = tokenizer([sentence], return_tensors='pt').to(DEVICE)

# pass it through the model
res = model(**tokd).logits

# argmax
classified_targets = torch.argmax(res, dim=2).cpu()

# 0, normal word
# 1, first capital
# 2, period
# 3, question mark
# 4, exclaimation mark
# 5, comma

# get the words from the sentence
tokenized_result = tokenizer.tokenize(sentence)
labeled_result = list(zip(tokenized_result, classified_targets[0].tolist()[1:-1]))

# and finally append the result
res_toks = []

# for each word, perform the action
for word, action in labeled_result:
    # set the working variable
    w = word

    # perform the edit actions
    if action == 1:
        w[0] = w[0].upper()
    elif action == 2:
        w = w+'.'
    elif action == 3:
        w = w+'?'
    elif action == 4:
        w = w+'!'
    elif action == 5:
        w = w+','

    # append
    res_toks.append(w)

# compose final passage
final_passage = tokenizer.convert_tokens_to_string(res_toks)
split_passage = sent_tokenize(final_passage)

split_passage
