"""Build the submission cover and technical report from actual saved results.
Requires python-docx; use Word/LibreOffice to export the resulting DOCX files to PDF.
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
REPO='https://github.com/KothariMayaank/Fgai_Assgn_2/tree/main/assignment3'
TITLE='English to French Translation with Transformer mT5 LLaMA and Gemma'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def style(d):
    s=d.sections[0];s.top_margin=Inches(.7);s.bottom_margin=Inches(.65);s.left_margin=Inches(.75);s.right_margin=Inches(.75)
    s.page_height=Inches(11.69);s.page_width=Inches(8.27)
    normal=d.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5)
    normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.line_spacing=1.08
    for n,size in [('Title',23),('Heading 1',16),('Heading 2',12)]:
        d.styles[n].font.name='Calibri';d.styles[n].font.size=Pt(size);d.styles[n].font.color.rgb=RGBColor(0,0,0)
    for border in list(d.styles.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
    d.styles['Subtitle'].font.color.rgb=RGBColor(70,70,70)
    d.core_properties.author='Mayaank Kothari'

def link(d,text,url):
    p=d.add_paragraph();h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True))
    r=OxmlElement('w:r');rp=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'245B80');rp.append(c);r.append(rp)
    t=OxmlElement('w:t');t.text=text;r.append(t);h.append(r);p._p.append(h)

def table(d,headers,rows):
    t=d.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1'
    for c,txt in zip(t.rows[0].cells,headers):c.text=txt
    for row in rows:
        for c,txt in zip(t.add_row().cells,row):c.text=str(txt)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(4)
                for r in p.runs:
                    r.font.size=Pt(9);r.font.color.rgb=RGBColor(0,0,0)
    return t

def main():
    rows=[]
    for name in ['basic','mt5','llama','gemma']:
        p=ROOT/f'results/{name}.json';rows.append(read(p) if p.exists() else {'model':name,'status':'not_run'})
    complete=[r for r in rows if r['status']=='completed'];pending=[r['model'] for r in rows if r['status']!='completed']
    manifest=read(ROOT/'data/manifest.json')
    d=Document();style(d)
    d.add_heading('Foundations of Generative AI',0)
    d.add_paragraph('ICT4403 | Assignment 3',style='Subtitle')
    for k,v in [('Student name','Mayaank Kothari'),('Registration number','230953562'),
      ('Assignment title','Comparative Machine Translation and Responsible AI'),
      ('System title',TITLE),('Language direction','English to French'),
      ('Dataset','Tatoeba English–French sentence pairs from the PyTorch tutorial snapshot')]:
        p=d.add_paragraph();p.add_run(k+'\n').bold=True;p.add_run(v)
    d.add_paragraph('Dataset source',style='Heading 2')
    link(d,'https://download.pytorch.org/tutorial/data.zip','https://download.pytorch.org/tutorial/data.zip')
    link(d,'Original dataset source: ManyThings / Tatoeba','https://www.manythings.org/anki/')
    d.add_paragraph('GitHub repository',style='Heading 2');link(d,REPO,REPO)
    if pending:d.add_paragraph('Experiment status: completed runs and remaining model-access requirements are documented in the repository report.')
    d.save(ROOT/'Mayaank_Kothari_230953562_FGAI_Assignment3.docx')

    d=Document();style(d)
    d.add_heading('Comparing English to French Translation Models',0)
    d.add_paragraph('ICT4403 Assignment 3 | Mayaank Kothari | 230953562',style='Subtitle')
    d.add_paragraph('This study implements a scratch Transformer, fine-tuned mT5, and prompted LLaMA and Gemma translation systems. Its purpose is to compare translation quality and computational cost on a shared test set, while examining failures relevant to responsible AI.')
    if pending:d.add_paragraph('The experiment is incomplete: '+', '.join(pending)+' have not produced completed test results. No four-model ranking or best-model conclusion is supported yet. The repository supplies executable paths for all four models and records available measurements without substituting estimated scores.')
    else:d.add_paragraph('All four systems have completed evaluation. The results below apply to this small dataset subset and execution environment; they do not establish an architecture-wide advantage.')
    d.add_heading('Dataset and preprocessing',1)
    d.add_paragraph(f"The public PyTorch tutorial archive contains {manifest['archive_rows']:,} English–French pairs originating from Tatoeba through ManyThings. After length filtering and exact case-insensitive source/target deduplication, {manifest['eligible_unique_rows']:,} pairs remain. A seed-42 shuffle selects 1,024 training, 128 validation and 128 test pairs. Alternative translations sharing the same source are removed; evaluation uses one reference per sentence.")
    d.add_paragraph('Text is normalized with Unicode NFKC and whitespace collapse while retaining accents. Source sentences contain 2–12 regex tokens; references contain 2–16. The scratch model lowercases and builds separate vocabularies from training data only. Pretrained models retain case and use their own tokenizers. Exact source and target overlap across splits is checked. Semantic near-duplicates and overlap with pretraining cannot be excluded.')
    d.add_heading('Model implementation',1)
    table(d,['System','Architecture and adaptation'],[
      ['Basic Transformer','2 encoder and 2 decoder layers; hidden size 128; 4 attention heads; FFN 256; trained from scratch.'],
      ['mT5 small','Pretrained encoder-decoder; full fine-tuning using translation-prefixed input and French labels.'],
      ['LLaMA 3.2 1B Instruct','Decoder-only; three-shot prompting with the first three training pairs and native chat template.'],
      ['Gemma 3 1B IT','Decoder-only; identical translation instruction and demonstrations, with its native chat template.']])
    d.add_paragraph('The scratch model uses a causal target attention mask, padding masks, teacher forcing and ignored padding loss. The same target restriction prevents future output tokens from leaking into training predictions. Prompted models receive source text as data, with instructions to preserve names, numbers, negation and tone.')
    d.add_page_break()
    d.add_heading('Evaluation and measured results',0)
    d.add_paragraph('Primary evaluation uses the same 128 test sources, greedy decoding, at most 48 generated tokens and inference batch size 1. A validation warm-up is excluded. Timings include tokenization, generation and decoding, but exclude model loading. Training time includes validation and checkpoint saving. SacreBLEU computes corpus BLEU with 13a tokenization and chrF with character order 6 and beta 2. Both scores range from 0 to 100.')
    def number(r,k):return f'{r[k]:.2f}' if isinstance(r.get(k),(int,float)) else 'N/A'
    table(d,['Model','Status','BLEU','chrF','Train s','Infer s'],[[r['model'],r['status'],number(r,'bleu'),number(r,'chrf'),number(r,'training_seconds'),number(r,'inference_seconds')] for r in rows])
    d.add_paragraph('N/A denotes an unavailable measurement or non-applicable task-specific training, never a measured zero. Prompting does not erase the cost of original pretraining. Exact parameter counts, metric signatures, software versions and test hashes are retained in results JSON files.')
    if complete:
        d.add_paragraph('Measured parameter counts: '+ '; '.join(f"{r['model']}: {r['parameters']:,}" for r in complete)+'.')
        if (ROOT/'results/quality.png').exists():d.add_picture(str(ROOT/'results/quality.png'),width=Inches(5.9))
    d.add_heading('Training choices and interpretation',1)
    d.add_paragraph('The scratch model uses AdamW at 0.0005, batch size 32, dropout 0.1 and 15 epochs. mT5 uses Adafactor at 0.001, batch size 4 and 2 epochs, with gradient checkpointing. Each selects the lowest validation token loss checkpoint; test scores do not choose checkpoints. The single seed and small subset limit statistical reliability. Poor outputs should motivate more data and validation tuning rather than hiding failures.')
    d.add_paragraph('The comparison confounds architecture with size, pretraining, tokenization and adaptation. Lowercasing also disadvantages the scratch model in case-sensitive scoring. Equal output-token caps have different character budgets. Timing across CPU/GPU or precision settings must not be treated as a controlled architecture comparison. A large pretrained model can exploit broad language knowledge unavailable to the scratch model.')
    d.add_page_break()
    d.add_heading('Translation errors and responsible AI',0)
    d.add_paragraph('The following examples are taken directly from the saved test predictions of the basic Transformer. They demonstrate errors in this run, not a general property of every Transformer. Outputs are quoted exactly except where an explicitly labelled excerpt is used.')
    p=ROOT/'results/basic_translations.json'
    examples=read(p) if p.exists() else []
    for i in [0,2,3]:
        if len(examples)<=i:continue
        r=examples[i];d.add_heading('Test example '+str(i+1),2)
        d.add_paragraph('Source: '+r['source']+'\nReference: '+r['reference'])
        out=r['translation'];suffix=' [excerpt]' if len(out)>160 else ''
        d.add_paragraph('Generated: '+out[:160]+suffix)
    d.add_heading('Risk assessment and mitigations',1)
    d.add_paragraph('Omissions and unsupported content: a fluent but incomplete translation can change the user’s meaning. In the first example, the property “weak” is lost. The age statement illustrates repetition and failure to preserve the number 23. Check named entities, quantities and negation against the source; route uncertain or high-stakes translations to a qualified bilingual reviewer.')
    d.add_paragraph('Bias and cultural meaning: English gender-neutral roles can become gendered in French, while festivals and idioms can be flattened into stereotypes or literal wording. Evaluate balanced occupations and genders, retain culturally specific names, and use bilingual review and terminology glossaries. These are prospective risks; unexecuted model probes are not evidence of observed bias.')
    d.add_paragraph('Sensitive content and hallucination: translations should preserve attribution, uncertainty and severity without adding accusations, medical facts or identifying details. Avoid automatically treating refusal as correctness or silent sanitization as faithful translation. The diagnostic probe set includes threats, medical negation, names/numbers and instruction-like source text. Probe outputs require human judgment and are excluded from headline BLEU/chrF.')
    d.add_paragraph('Privacy: use public corpus data for this project. Do not publish private translation inputs or authentication tokens. Automatic metrics measure reference overlap, not cultural competence, factual safety or permission to deploy.')
    d.add_page_break()
    d.add_heading('Reproducibility and completion',0)
    d.add_heading('Temperature analysis',1)
    d.add_paragraph('The main translation comparison is greedy. The auxiliary decoder-only sweep samples at temperatures 0.3, 0.7 and 1.0, with top-p 0.9 and seeds 42, 43 and 44 for each temperature. The same held-out test set is used; all generated outputs and corpus scores are saved. This addresses the additional README checklist item without mixing sampled scores into the primary comparison.')
    sweeps=[name for name in ['llama','gemma'] if (ROOT/f'results/{name}_temperature_summary.json').exists()]
    if not sweeps:d.add_paragraph('No temperature sweep has completed in the supplied run. A possible diversity–fidelity trade-off is a hypothesis, not an observed finding. Run the notebook sweep and review factual errors, repetitions and score variability before drawing a conclusion.')
    else:
        for name in sweeps:
            rs=read(ROOT/f'results/{name}_temperature_summary.json')
            table(d,['Model','Temp','Mean BLEU','Mean chrF'],[[name,str(t),f"{sum(r['bleu'] for r in rs if r['temperature']==t)/3:.2f}",f"{sum(r['chrf'] for r in rs if r['temperature']==t)/3:.2f}"] for t in [.3,.7,1.0]])
    d.add_heading('Execution status and next steps',1)
    d.add_paragraph('Run the supplied Colab notebook on a GPU after obtaining model access from Hugging Face. LLaMA and Gemma downloads require the account holder to accept their respective terms and authenticate within the notebook. No model-access gate is bypassed. Rerun all models on one runtime for comparable timing. Regenerate the comparison and documents after the remaining runs complete.')
    d.add_paragraph('Four automated checks passed in the supplied local run: split separation, future-token masking, exact-match metric sanity, and vocabulary restricted to training data. Those checks validate specific properties; they do not validate unavailable models or prove translation quality. The source archive hash, exact splits, model IDs and measured environment are stored with the project.')
    d.add_heading('Conclusion',1)
    if pending:d.add_paragraph('The project implements the required model paths and documents available experimental evidence. The small scratch model exhibits low reference overlap and semantic failures. A four-model winner and a completed architecture comparison cannot be reported until all required runs finish. The remaining work is model execution and bilingual review, not replacement of missing scores with estimates.')
    else:
        winner=max(complete,key=lambda r:r['chrf'])
        d.add_paragraph(f"{winner['model']} has the highest chrF in this run. Consider its BLEU, latency, size and qualitative errors before selecting it. The conclusion is restricted to this subset and setup.")
    d.add_heading('Sources and project',1)
    for text,url in [('Dataset and attribution','https://www.manythings.org/anki/'),('Dataset snapshot','https://download.pytorch.org/tutorial/data.zip'),('mT5 model card','https://huggingface.co/google/mt5-small'),('LLaMA model card','https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct'),('Gemma model card','https://huggingface.co/google/gemma-3-1b-it'),('SacreBLEU evaluation implementation','https://github.com/mjpost/sacrebleu'),('Project repository',REPO)]:link(d,text+' — '+url,url)
    d.save(ROOT/'Assignment3_Report.docx')
    # Keep a simple Markdown report alongside the Word version for GitHub review.
    lines=[]
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    for elem in d.element.body:
        if elem.tag==qn('w:p'):
            para=Paragraph(elem,d)
            text=''.join(elem.xpath('.//w:t/text()'))
            if text.strip():lines.append(('## ' if para.style.name.startswith('Heading') else '')+text)
        elif elem.tag==qn('w:tbl'):
            t=Table(elem,d);data=[[c.text.replace('\n',' ') for c in r.cells] for r in t.rows]
            lines.append('\n'.join(['| '+' | '.join(data[0])+' |','| '+' | '.join(['---']*len(data[0]))+' |']+['| '+' | '.join(r)+' |' for r in data[1:]]))
    (ROOT/'REPORT.md').write_text('\n\n'.join(lines)+'\n\nSee results/RESULTS.md for the numerical comparison.\n',encoding='utf-8')
    print('Created cover DOCX, report DOCX and REPORT.md')

if __name__=='__main__':main()
