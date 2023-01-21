import torch
import torch.nn as nn
import json
from trainer import train,eval
from cost import crit_weights_gen
from net import Net
from dataset import NerDataset, pad, VOCAB, tag2idx, idx2tag
import dataset
import torch.optim as optim
import os

# dataset.VOCAB = ('<PAD>', 'O',
# 'I-GRP',
# 'B-PROD',
# 'I-PER',
# 'I-CW',
# 'I-CORP',
# 'B-PER',
# 'B-CORP',
# 'B-GRP',
# 'B-LOC',
# 'B-CW',
# 'I-PROD',
# 'I-LOC')
#
# dataset.tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# dataset.idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

print(dataset.VOCAB)

batch_size = 32
lr = 0.001
n_epochs = 20
finetuning = True
top_rnns = True
trainset = "data/validation.jsonl"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(top_rnns, len(VOCAB), device, finetuning)
#model.load_state_dict(torch.load('models/banner_model.pt'))
model.to(device)

# with open(trainset) as infile:
#     data = json.load(infile)

data = []
for line in open(trainset, 'r', encoding="utf8"):
    data.append(json.loads(line))

new = data
train_texts, train_labels = list(zip(*map(lambda d: (d['tokens'], d['tags']), new)))

sents_train, tags_li_train = [], []
for x in train_texts:
    sents_train.append(["[CLS]"] + x + ["[SEP]"])
for y in train_labels:
    tags_li_train.append(["<PAD>"] + y + ["<PAD>"])

train_dataset = NerDataset(sents_train, tags_li_train)

train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                             batch_size= batch_size,
                             shuffle=True,
                             collate_fn=pad,
                             num_workers=0
                             )

optimizer = optim.Adam(model.parameters(), lr = lr)
# data_dist = [7237, 15684, 714867, 759, 20815, 9662, 8512, 37529, 70025]
# crit_weights = crit_weights_gen(0.5,0.9,data_dist)
#insert 0 cost for ignoring <PAD>
# crit_weights.insert(0,0)
# crit_weights = torch.tensor(crit_weights).to(device)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, n_epochs+1):
    if epoch>10:
        optimizer = optim.Adam([
                                {"params": model.fc.parameters(), "lr": 0.0005},
                                {"params": model.bert.parameters(), "lr": 5e-5},
                                {"params": model.rnn.parameters(), "lr": 0.0005},
                                {"params": model.crf.parameters(), "lr": 0.0005}
                                ],)
    train(model, train_iter, optimizer, criterion, epoch)
    # _ = eval(model, test_iter, epoch)

    fname = os.path.join("models", str(epoch))
    torch.save(model.state_dict(), f"{fname}.pt")