# ICT4403 Foundations of Generative AI Assignment 3

**Student:** Mayaank Kothari | **Registration:** 230953562

English-to-French machine translation with a basic Transformer, mT5, LLaMA and Gemma. This folder is independent of Assignment 2 at the repository root.

## Current status

See [results/RESULTS.md](results/RESULTS.md) for measured results and per-model execution status. The basic Transformer has been trained and evaluated. LLaMA and Gemma require approved Hugging Face access; their scores must not be inferred from other models. An incomplete comparison does not identify a four-model winner. The submission document is a cover sheet, not evidence that every experiment has completed.

## Run in Google Colab

Open [Assignment3.ipynb](Assignment3.ipynb) using [Google Colab](https://colab.research.google.com/github/KothariMayaank/Fgai_Assgn_2/blob/main/assignment3/Assignment3.ipynb). Select a GPU runtime. Follow the notebook cells, including account approval for the two gated models. The notebook uses the checked-in splits, trains the encoder-decoder models and prompts both decoder-only models. It exports results and regenerates the report. Do not publish tokens or cached model weights.

## Local execution

Python 3.11 or 3.12 is recommended. Install the appropriate PyTorch build for your hardware first if needed.

```bash
cd assignment3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Use the committed data splits for comparison with the supplied run.
python -m unittest test_pipeline -v
python experiment.py basic --epochs 15
python experiment.py mt5 --epochs 2
# Before the next two commands: accept the model terms and authenticate locally.
hf auth login
python experiment.py llama --temperatures
python experiment.py gemma --temperatures
python experiment.py summarize
python inspect_checkpoints.py basic
python inspect_checkpoints.py mt5
python make_documents.py
```

Windows activation: `.venv\Scripts\activate`. `HF_TOKEN` may be supplied as a private environment variable or Colab secret; never paste it into a report or commit it. The program uses the standard Hugging Face credential store.

For a longer experiment, regenerate all splits with `python experiment.py prepare --train-size 10000 --validation-size 1000 --test-size 1000`, then rerun **all four models**. Do not mix scores from different test fingerprints. Regenerating overwrites the local splits. Results are overwritten only for the model being run.

## Dataset and source

- Language direction: English to French.
- Original corpus: [Tatoeba sentence pairs distributed by ManyThings](https://www.manythings.org/anki/).
- Exact downloadable snapshot: [PyTorch tutorial data archive](https://download.pytorch.org/tutorial/data.zip), member `data/eng-fra.txt`.
- Dataset context: [official PyTorch translation tutorial](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).
- The snapshot contains 135,842 rows. The preprocessing retains 82,935 eligible pairs before random subset selection.
- Training / validation / test: **1,024 / 128 / 128**, seed 42. This resource-limited teaching subset is not a benchmark of general translation quality.
- Normalize Unicode to NFKC and collapse whitespace. Keep accents. Keep source lengths 2–12 and target lengths 2–16 regex tokens. Remove repeated case-insensitive source or target sentences before splitting; retain the first pair. This discards alternative references rather than evaluating multiple references.
- Use case-preserving text for pretrained models. The scratch model alone lowercases and uses train-only word/punctuation vocabularies. Test unknown words map to `<unk>`.
- Split IDs and archive/split hashes are recorded in `data/manifest.json`. The source archive is not committed. The snapshot lacks per-sentence author fields; see `DATA_ATTRIBUTION.md` for provenance and limitations.

## Models and experimental design

| Model | Architecture | Adaptation |
|---|---|---|
| Basic Transformer | 2 encoder + 2 decoder layers, 128 hidden units, 4 heads, FFN 256 | Train from scratch, AdamW, 15 epochs |
| `google/mt5-small` | Pretrained encoder-decoder | Full fine-tuning, Adafactor, 2 epochs |
| `meta-llama/Llama-3.2-1B-Instruct` | Pretrained decoder-only | Three-shot prompting using only the first 3 training pairs |
| `google/gemma-3-1b-it` | Pretrained decoder-only | Same three-shot task and examples |

Basic training: batch 32, learning rate 0.0005, dropout 0.1, gradient clipping 1.0. mT5: batch 4, learning rate 0.001, gradient checkpointing; pretrained weights are updated. Validation token loss chooses the checkpoint; test data never chooses epochs or prompts. No paid inference service is used.

All primary runs use the identical test split, greedy decoding, 48 new tokens maximum, inference batch size 1, and a validation warm-up excluded from timing. Inference timing includes tokenization, generation and decoding; downloads/loading are excluded and separately recorded. CUDA synchronization is used where available. Report full-corpus SacreBLEU BLEU (`13a`) and chrF (character order 6, beta 2), including signatures. Parameter counts come from the loaded model. Prompted models have **N/A** task training time, not zero pretraining cost.

Comparability is limited: model sizes, multilingual pretraining, tokenizers, lowercasing, adaptation method, effective prompt lengths and training budgets differ. Equal token caps do not mean equal character budgets. These results cannot isolate architecture as a causal factor. Public benchmark contamination and near-duplicate leakage remain possible despite exact deduplication. Compare timing only on the same hardware/dtype.

## Results and model comparison

The measured score table is [results/RESULTS.md](results/RESULTS.md). `results/*_translations.json` contains every scored example, including unwanted commentary and empty output; no favourable examples are substituted. Loss histories, environment metadata, parameter counts and split hashes accompany each successful run. `results/quality.png` shows completed models only. Failed or blocked runs have no numerical score.

`REPORT.md` and the PDF report explain the actual available findings, architectural trade-offs and responsible-use concerns. A best model is identified only after all four models finish on the same test set; even then it is specific to this dataset and compute budget.

## Sampling-temperature analysis

The assignment body concerns translation, although the submission checklist also asks for temperature analysis. The main comparison is deterministic. `--temperatures` adds an auxiliary sweep for the two prompted models at **0.3, 0.7 and 1.0**, each with seeds **42, 43 and 44**, top-p 0.9, on the same test set. It records BLEU, chrF, runtime and all outputs. This must be completed before claiming an observed temperature effect. In expectation, increased sampling can diversify wording but may hurt factual fidelity; that hypothesis is not a measured result. The report distinguishes completed and pending sweeps.

## Responsible translation

Inspect actual test outputs for omissions, unsupported details, gender assumptions, altered names/numbers/negation, literal idioms, cultural distortion and mishandled sensitive statements. `ethical_probes.json` adds targeted diagnostic examples; these are not training data and are excluded from headline scores. Probe files, when generated, require bilingual human review; automated scores alone cannot establish safety.

Use diverse reviewed training data, content-as-data prompts, entity/number/negation checks, a terminology glossary and bilingual review. Escalate medical/legal or other high-stakes content to qualified translators. Do not treat fluency, a refusal, or a high BLEU score as proof of semantic accuracy or safety. Preserve the meaning of sensitive content without embellishment or silent censorship, subject to the intended use and applicable policy. Store only necessary source text and remove private data before public sharing.

## Files and submission

- `experiment.py`: data preparation, all four model paths, metrics, timings, temperature sweeps and plotting.
- `test_pipeline.py`: checks split leakage, target causality, metrics and vocabulary leakage.
- `inspect_checkpoints.py`: reproduce targeted diagnostic outputs from saved encoder–decoder checkpoints.
- `Assignment3.ipynb`: Colab execution flow.
- `data/`: exact train/validation/test split plus provenance manifest.
- `results/`: actual outputs and metrics.
- `make_documents.py`: regenerate the report and cover DOCX from saved results.
- `REPORT.md`: experimental analysis and completion status.
- `Mayaank_Kothari_230953562_FGAI_Assignment3.docx` and `.pdf`: required cover document.
- `Assignment3_Report.pdf`: technical report.

The cover uses the translation-system title because the instructions' reference to an autoregressive text-generation system appears copied from Assignment 2. Upload the cover PDF to the college LMS only after reviewing the repository and completion status. LMS upload is not performed by this project.
