"""Reproduce targeted ethical probe outputs from trained local checkpoints.
These outputs require bilingual human review and are excluded from test metrics.
"""
import argparse,re
import torch
import experiment as e

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['basic','mt5']);args=p.parse_args()
    device=e.setup();probes=e.read(e.ROOT/'ethical_probes.json')
    if args.model=='basic':
        state=torch.load(e.ROOT/'checkpoints/basic.pt',map_location=device,weights_only=True)
        sv,tv=state['src_vocab'],state['tgt_vocab'];inv={v:k for k,v in tv.items()}
        model=e.make_basic(len(sv),len(tv)).to(device);model.load_state_dict(state['state']);model.eval()
        @torch.inference_mode()
        def translate(text):
            s=torch.tensor([[1]+[sv.get(w,3) for w in e.tokens(text)]+[2]],device=device)
            mem=model.encode(s);t=torch.tensor([[1]],device=device)
            for _ in range(48):
                logits=model.decode(s,mem,t)[:,-1,:];logits[:,0:2]=-float('inf')
                nxt=logits.argmax(-1).view(1,1);t=torch.cat([t,nxt],1)
                if nxt.item()==2:break
            return re.sub(r'\s+([.,!?;:])',r'\1',' '.join(inv[i] for i in t[0,1:].tolist() if i!=2))
    else:
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        cp=e.ROOT/'checkpoints/mt5';tok=AutoTokenizer.from_pretrained(cp)
        model=AutoModelForSeq2SeqLM.from_pretrained(cp).to(device).eval();model.config.use_cache=True
        @torch.inference_mode()
        def translate(text):
            x=tok('translate English to French: '+text,return_tensors='pt',max_length=64,truncation=True).to(device)
            return tok.decode(model.generate(**x,max_new_tokens=48,do_sample=False,num_beams=1)[0],skip_special_tokens=True).strip()
    e.save(e.ROOT/f'results/{args.model}_ethical_probes.json',
           [{**r,'translation':translate(r['source']),'human_review':'pending'} for r in probes])

if __name__=='__main__':main()
