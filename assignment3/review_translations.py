"""Create a diagnostic review worksheet from saved outputs.
The heuristic flags are review aids, not automatic semantic/safety judgments.
"""
import collections,re
from pathlib import Path
import experiment as e

def flags(source,pred):
    out=[]
    # French may spell a number out, so this is only a potential mismatch flag.
    if re.findall(r'\d+',source)!=re.findall(r'\d+',pred):out.append('Check numbers (digits differ)')
    words=e.tokens(pred)
    if len(words)>6:
        bigrams=list(zip(words,words[1:]))
        if max(collections.Counter(bigrams).values(),default=0)>=3:out.append('Repeated bigram >=3 times')
    if not pred.strip():out.append('Empty output')
    if len(words)>=48:out.append('Check possible length-cap truncation')
    return '; '.join(out) or 'No heuristic flag; human review still required'

def main():
    lines=['# Translation review','','These tables contain actual saved outputs. Flags are heuristic review prompts, not confirmed errors. The first five test examples are selected by fixed split order, not by score.','','## Shared test examples','']
    for name in ['basic','mt5','llama','gemma']:
        path=e.ROOT/f'results/{name}_translations.json'
        lines+=['### '+name,'']
        if not path.exists():lines+=['No completed translations available.',''];continue
        data=e.read(path)
        lines+=['| ID | Source | Reference | Output | Review prompt |','|---|---|---|---|---|']
        for r in data[:5]:
            vals=[r['id'],r['source'],r['reference'],r['translation'],flags(r['source'],r['translation'])]
            lines.append('| '+' | '.join(x.replace('|','\\|').replace('\n','<br>') for x in vals)+' |')
        lines.append('')
    lines+=['## Targeted diagnostic probes','','These additional sources are excluded from BLEU and chrF test scores. Findings require bilingual review.','']
    for name in ['basic','mt5','llama','gemma']:
        path=e.ROOT/f'results/{name}_ethical_probes.json'
        if not path.exists():continue
        lines+=['### '+name,'','| Concern | Source | Actual output | Review criterion |','|---|---|---|---|']
        for r in e.read(path):
            vals=[r['category'],r['source'],r['translation'],r['review_criterion']]
            lines.append('| '+' | '.join(x.replace('|','\\|').replace('\n','<br>') for x in vals)+' |')
        lines.append('')
    (e.ROOT/'results/QUALITATIVE_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
