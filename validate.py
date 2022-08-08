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

# import our dataset
from dataset import TOKENS, BOUNDARIES, UtteranceBoundaryDataset, calculate_acc_prec_rec_f1

# set model
MODEL = "./models/flowing-salad-6"
DATA = "./data/Pitt.txt"

# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the model
model = BertForTokenClassification.from_pretrained(MODEL,
                                                   num_labels=len(TOKENS)).to(DEVICE)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# create data collator utility on the tokenizer
data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

# load the validation dataset
dataset = DataLoader(UtteranceBoundaryDataset(DATA, tokenizer, 20),
                     batch_size=4,
                     shuffle=True,
                     collate_fn=lambda x:x)

# get a list going
accuracies = []
precisions = []
recalls = []
f1s = []

for sample in tqdm(iter(dataset)):
    # get the batch collected
    batch = data_collator(sample)

    # get the model output
    output = model(**batch)
    preds = torch.argmax(output.logits, dim=2)

    # calculate metrics
    acc, prec, recc, f1 = calculate_acc_prec_rec_f1(preds.cpu().detach(),
                                                    batch["labels"].cpu().detach())

    # append results
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(recc)
    f1s.append(f1)

# append results
mean_acc = sum(accuracies)/len(accuracies)
mean_precisions = sum(precisions)/len(precisions)
mean_recalls = sum(recalls)/len(recalls)
mean_f1s = sum(f1s)/len(f1s)

mean_acc
mean_precisions
mean_recalls
mean_f1s




