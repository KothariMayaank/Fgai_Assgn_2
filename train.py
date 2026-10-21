"""Reproducible word-level recurrent language-model comparison (CPU default)."""
import argparse
import collections
import csv
import hashlib
import json
import math
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODELS = ['RNN', 'BiRNN', 'LSTM', 'GRU']
SEEDS = ['alice was beginning to get very tired', 'the white rabbit', 'she looked at the']
TEMPERATURES = [0.5, 1.0, 1.5]

def tokenize(text):
    # Word-level task: punctuation is deliberately discarded; contractions retained.
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace('\u2019', "'"))

def prepare(context):
    raw = (ROOT / 'data/alice.txt').read_text(encoding='utf-8-sig')
    body = re.split(r'\*\*\* START OF .*?\*\*\*', raw, maxsplit=1)[1]
    body = re.split(r'\*\*\* END OF .*?\*\*\*', body, maxsplit=1)[0]
    # Remove title, illustration credits and contents before actual Chapter I.
    start = body.index('Alice was beginning')
    tokens = tokenize(body[start:])
    a, b = int(len(tokens)*0.8), int(len(tokens)*0.9)
    splits = [tokens[:a], tokens[a:b], tokens[b:]]
    counts = collections.Counter(splits[0])
    vocab = ['<unk>'] + sorted(w for w,c in counts.items() if c >= 2)
    stoi = {w:i for i,w in enumerate(vocab)}
    datasets, stats = [], {}
    for name, words in zip(['train','validation','test'], splits):
        ids = torch.tensor([stoi.get(w,0) for w in words], dtype=torch.long)
        windows = ids.unfold(0, context+1, 1)
        datasets.append(TensorDataset(windows[:,:-1].contiguous(), windows[:,-1].contiguous()))
        stats[name] = dict(words=len(words), sequences=len(windows), unknown_fraction=float((ids==0).float().mean()))
    stats.update(total_words=len(tokens), raw_unique_words=len(set(tokens)), vocabulary=len(vocab), dataset_sha256=hashlib.sha256(raw.encode()).hexdigest())
    return datasets, vocab, stats

class LanguageModel(nn.Module):
    def __init__(self, kind, vocabulary, embedding=64, hidden=96):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding)
        self.kind = kind
        recurrent = {'RNN':nn.RNN, 'BiRNN':nn.RNN, 'LSTM':nn.LSTM, 'GRU':nn.GRU}[kind]
        self.recurrent = recurrent(embedding, hidden, batch_first=True, bidirectional=kind=='BiRNN')
        self.output = nn.Linear(hidden*(2 if kind=='BiRNN' else 1), vocabulary)

    def forward(self, x):
        _, state = self.recurrent(self.embedding(x))
        if self.kind == 'LSTM':
            state = state[0]
        # Both final directional states summarize only the supplied prefix.
        h = torch.cat([state[-2],state[-1]],dim=-1) if self.kind=='BiRNN' else state[-1]
        return self.output(h)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss, correct, n = 0., 0, 0
    for x,y in loader:
        z = model(x)
        loss += nn.functional.cross_entropy(z,y,reduction='sum').item()
        correct += (z.argmax(-1)==y).sum().item()
        n += len(y)
    return dict(loss=loss/n, perplexity=math.exp(loss/n), accuracy=correct/n)

@torch.no_grad()
def generate(model, vocab, seed, temperature, context, length=50, random_seed=123):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    model.eval()
    stoi = {w:i for i,w in enumerate(vocab)}
    ids = [stoi.get(w,0) for w in tokenize(seed)]
    if not ids:
        raise ValueError('Seed must contain a word')
    rng = torch.Generator().manual_seed(random_seed)
    words = []
    for _ in range(length):
        logits = model(torch.tensor([ids[-context:]],dtype=torch.long))[0] / temperature
        logits[0] = -torch.inf  # Do not print unknown tokens; evaluation remains unmasked.
        index = torch.multinomial(logits.softmax(-1),1,generator=rng).item()
        ids.append(index)
        words.append(vocab[index])
    pairs = list(zip(words,words[1:]))
    return dict(text=seed+' '+' '.join(words), distinct1=len(set(words))/len(words), distinct2=len(set(pairs))/len(pairs), repetition=1-len(set(pairs))/len(pairs))

def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--context',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if min(args.epochs,args.context,args.batch_size,args.threads)<1:
        p.error('Epochs, context, batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    datasets,vocab,stats=prepare(args.context)
    result=ROOT/'results'; result.mkdir(exist_ok=True)
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    (result/'dataset.json').write_text(json.dumps(stats,indent=2))
    config=vars(args)|dict(embedding=64,hidden=96,learning_rate=0.003,optimizer='Adam',gradient_clip=1.0,torch=torch.__version__,python=platform.python_version(),platform=platform.platform(),device='cpu')
    (result/'config.json').write_text(json.dumps(config,indent=2))
    (result/'vocab.json').write_text(json.dumps(vocab))
    all_history, comparison, samples=[],[],[]
    for kind in MODELS:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model=LanguageModel(kind,len(vocab))
        optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
        train=DataLoader(datasets[0],batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        validation=DataLoader(datasets[1],batch_size=256)
        test=DataLoader(datasets[2],batch_size=256)
        best=float('inf'); best_epoch=0; train_seconds=0.
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; n=0; start=time.perf_counter()
            for x,y in train:
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x),y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                total+=loss.item()*len(y); n+=len(y)
            elapsed=time.perf_counter()-start; train_seconds+=elapsed
            metrics=evaluate(model,validation)
            all_history.append(dict(model=kind,epoch=epoch,train_loss=total/n,validation_loss=metrics['loss'],validation_accuracy=metrics['accuracy'],training_seconds=elapsed))
            if metrics['loss']<best:
                best=metrics['loss']; best_epoch=epoch
                torch.save(dict(state_dict=model.state_dict(),kind=kind,vocab=vocab,context=args.context),ROOT/'checkpoints'/f'{kind}.pt')
            print(f'{kind} {epoch:02d}: train={total/n:.4f} val={metrics["loss"]:.4f} time={elapsed:.1f}s',flush=True)
        checkpoint=torch.load(ROOT/'checkpoints'/f'{kind}.pt',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        comparison.append(dict(model=kind,parameters=sum(t.numel() for t in model.parameters()),best_epoch=best_epoch,validation_loss=best,training_seconds=train_seconds,**evaluate(model,test)))
        for seed_i,seed in enumerate(SEEDS):
            for temperature in TEMPERATURES:
                for repeat in range(3):
                    random_seed=1000+100*seed_i+repeat
                    samples.append(dict(model=kind,seed=seed,temperature=temperature,replicate=repeat+1,**generate(model,vocab,seed,temperature,args.context,random_seed=random_seed)))
        save_csv(result/'comparison.csv',comparison)
        save_csv(result/'history.csv',all_history)
        save_csv(result/'samples.csv',samples)
    # Training-frequency unigram baseline, with add-one smoothing.
    train_targets=datasets[0].tensors[1]
    frequency=torch.bincount(train_targets,minlength=len(vocab)).float()+1
    prob=frequency/frequency.sum(); target=datasets[2].tensors[1]
    loss=-prob[target].log().mean().item()
    (result/'baseline.json').write_text(json.dumps(dict(loss=loss,perplexity=math.exp(loss),accuracy=float((target==prob.argmax()).float().mean())),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in MODELS:
        h=[r for r in all_history if r['model']==kind]
        axes[0].plot([r['epoch'] for r in h],[r['validation_loss'] for r in h],label=kind)
    axes[0].set(xlabel='Epoch',ylabel='Validation cross entropy',title='Checkpoint selection'); axes[0].legend()
    axes[1].bar(MODELS,[r['perplexity'] for r in comparison],color=['#3569a8','#64a5bd','#d99b39','#53a078'])
    axes[1].set(ylabel='Test perplexity',title='Best validation checkpoints')
    fig.tight_layout(); fig.savefig(result/'comparison.png',dpi=180); plt.close(fig)
    aggregate=[]
    for kind in MODELS:
        for temp in TEMPERATURES:
            rows=[r for r in samples if r['model']==kind and r['temperature']==temp]
            aggregate.append(dict(model=kind,temperature=temp,**{k:float(np.mean([r[k] for r in rows])) for k in ['distinct1','distinct2','repetition']}))
    save_csv(result/'temperature_summary.csv',aggregate)
    fig,ax=plt.subplots(figsize=(7,4))
    for kind in MODELS:
        rows=[r for r in aggregate if r['model']==kind]
        ax.plot(TEMPERATURES,[r['distinct2'] for r in rows],marker='o',label=kind)
    ax.set(xlabel='Temperature',ylabel='Mean distinct bigram ratio',title='Diversity across nine continuations per setting'); ax.legend()
    fig.tight_layout(); fig.savefig(result/'temperature.png',dpi=180); plt.close(fig)
    print('Finished. Results and checkpoints saved.',flush=True)

if __name__=='__main__':
    main()
