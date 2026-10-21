"""Generate from a trained checkpoint: python generate.py --model GRU --seed 'alice was'."""
import argparse
import torch
from train import ROOT, LanguageModel, generate

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['RNN','BiRNN','LSTM','GRU'],default='LSTM')
    p.add_argument('--seed',default='alice was beginning to get very tired')
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--words',type=int,default=50)
    p.add_argument('--random-seed',type=int,default=123)
    args=p.parse_args()
    if args.words < 2: p.error('--words must be at least 2')
    torch.set_num_threads(4)
    c=torch.load(ROOT/'checkpoints'/f'{args.model}.pt',map_location='cpu',weights_only=True)
    model=LanguageModel(c['kind'],len(c['vocab']))
    model.load_state_dict(c['state_dict'])
    print(generate(model,c['vocab'],args.seed,args.temperature,c['context'],args.words,args.random_seed)['text'])
