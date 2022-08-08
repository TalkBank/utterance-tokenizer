# system utilities
import re
import string
import random

# tokenization utilitise
from nltk import word_tokenize

# torch
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

# import huggingface utils
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification

# import our dataset
from dataset import TOKENS, UtteranceBoundaryDataset

# tqdm
from tqdm import tqdm

# wandb
import wandb

# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# weights and biases
hyperparametre_defaults = dict(
    learning_rate = 3.5e-5,
    batch_size = 5,
    epochs = 2,
    window = 20 
)

# start wandb
run = wandb.init(project='utok', entity='jemoka', config=hyperparametre_defaults, mode="disabled")
# run = wandb.init(project='utok', entity='jemoka', config=hyperparametre_defaults)

# set configuration
config = run.config

# create the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load the data (train using MICASE, test on Pitt)
train_data = UtteranceBoundaryDataset("./data/AphasiaBankEnglishProtocol.txt", tokenizer, window=config.window)
test_data = UtteranceBoundaryDataset("./data/Pitt.txt", tokenizer, window=config.window)

# create data collator utility on the tokenizer
data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

# load the data
train_dataloader = DataLoader(train_data,
                              batch_size=config.batch_size,
                              shuffle=True, collate_fn=lambda x:x)
test_dataloader = DataLoader(test_data,
                             batch_size=config.batch_size,
                             shuffle=True, collate_fn=lambda x:x)

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

# watch the model
run.watch(model)

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

