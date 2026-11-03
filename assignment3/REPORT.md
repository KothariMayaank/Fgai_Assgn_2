Comparing English to French Translation Models

ICT4403 Assignment 3 | Mayaank Kothari | 230953562

This study implements a scratch Transformer, fine-tuned mT5, and prompted LLaMA and Gemma translation systems. Its purpose is to compare translation quality and computational cost on a shared test set, while examining failures relevant to responsible AI.

The experiment is incomplete: mt5, llama, gemma have not produced completed test results. No four-model ranking or best-model conclusion is supported yet. The repository supplies executable paths for all four models and records available measurements without substituting estimated scores.

## Dataset and preprocessing

The public PyTorch tutorial archive contains 135,842 English–French pairs originating from Tatoeba through ManyThings. After length filtering and exact case-insensitive source/target deduplication, 82,935 pairs remain. A seed-42 shuffle selects 1,024 training, 128 validation and 128 test pairs. Alternative translations sharing the same source are removed; evaluation uses one reference per sentence.

Text is normalized with Unicode NFKC and whitespace collapse while retaining accents. Source sentences contain 2–12 regex tokens; references contain 2–16. The scratch model lowercases and builds separate vocabularies from training data only. Pretrained models retain case and use their own tokenizers. Exact source and target overlap across splits is checked. Semantic near-duplicates and overlap with pretraining cannot be excluded.

## Model implementation

| System | Architecture and adaptation |
| --- | --- |
| Basic Transformer | 2 encoder and 2 decoder layers; hidden size 128; 4 attention heads; FFN 256; trained from scratch. |
| mT5 small | Pretrained encoder-decoder; full fine-tuning using translation-prefixed input and French labels. |
| LLaMA 3.2 1B Instruct | Decoder-only; three-shot prompting with the first three training pairs and native chat template. |
| Gemma 3 1B IT | Decoder-only; identical translation instruction and demonstrations, with its native chat template. |

The scratch model uses a causal target attention mask, padding masks, teacher forcing and ignored padding loss. The same target restriction prevents future output tokens from leaking into training predictions. Prompted models receive source text as data, with instructions to preserve names, numbers, negation and tone.

Evaluation and measured results

Primary evaluation uses the same 128 test sources, greedy decoding, at most 48 generated tokens and inference batch size 1. A validation warm-up is excluded. Timings include tokenization, generation and decoding, but exclude model loading. Training time includes validation and checkpoint saving. SacreBLEU computes corpus BLEU with 13a tokenization and chrF with character order 6 and beta 2. Both scores range from 0 to 100.

| Model | Status | BLEU | chrF | Train s | Infer s |
| --- | --- | --- | --- | --- | --- |
| basic | completed | 0.31 | 11.19 | 31.46 | 1.46 |
| mt5 | not_run | N/A | N/A | N/A | N/A |
| llama | blocked_access | N/A | N/A | N/A | N/A |
| gemma | blocked_access | N/A | N/A | N/A | N/A |

N/A denotes an unavailable measurement or non-applicable task-specific training, never a measured zero. Prompting does not erase the cost of original pretraining. Exact parameter counts, metric signatures, software versions and test hashes are retained in results JSON files.

Measured parameter counts: basic: 1,299,157.

## Training choices and interpretation

The scratch model uses AdamW at 0.0005, batch size 32, dropout 0.1 and 15 epochs. mT5 uses Adafactor at 0.001, batch size 4 and 2 epochs, with gradient checkpointing. Each selects the lowest validation token loss checkpoint; test scores do not choose checkpoints. The single seed and small subset limit statistical reliability. Poor outputs should motivate more data and validation tuning rather than hiding failures.

The comparison confounds architecture with size, pretraining, tokenization and adaptation. Lowercasing also disadvantages the scratch model in case-sensitive scoring. Equal output-token caps have different character budgets. Timing across CPU/GPU or precision settings must not be treated as a controlled architecture comparison. A large pretrained model can exploit broad language knowledge unavailable to the scratch model.

Translation errors and responsible AI

The following examples are taken directly from the saved test predictions of the basic Transformer. They demonstrate errors in this run, not a general property of every Transformer. Outputs are quoted exactly except where an explicitly labelled excerpt is used.

## Test example 1

Source: Tom is weak.Reference: Tom est faible.

Generated: tom est.

## Test example 3

Source: They're arguing.Reference: Ils sont en train de se disputer.

Generated: ils sont sont sont sont.

## Test example 4

Source: When I painted this picture, I was 23 years old.Reference: Quand j'ai peint ce tableau, j'avais 23 ans.

Generated: je ne veux, je suis pas que je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je suis, je

## Risk assessment and mitigations

Omissions and unsupported content: a fluent but incomplete translation can change the user’s meaning. In the first example, the property “weak” is lost. The age statement illustrates repetition and failure to preserve the number 23. Check named entities, quantities and negation against the source; route uncertain or high-stakes translations to a qualified bilingual reviewer.

Bias and cultural meaning: English gender-neutral roles can become gendered in French, while festivals and idioms can be flattened into stereotypes or literal wording. Evaluate balanced occupations and genders, retain culturally specific names, and use bilingual review and terminology glossaries. These are prospective risks; unexecuted model probes are not evidence of observed bias.

Sensitive content and hallucination: translations should preserve attribution, uncertainty and severity without adding accusations, medical facts or identifying details. Avoid automatically treating refusal as correctness or silent sanitization as faithful translation. The diagnostic probe set includes threats, medical negation, names/numbers and instruction-like source text. Probe outputs require human judgment and are excluded from headline BLEU/chrF.

Privacy: use public corpus data for this project. Do not publish private translation inputs or authentication tokens. Automatic metrics measure reference overlap, not cultural competence, factual safety or permission to deploy.

Reproducibility and completion

## Temperature analysis

The main translation comparison is greedy. The auxiliary decoder-only sweep samples at temperatures 0.3, 0.7 and 1.0, with top-p 0.9 and seeds 42, 43 and 44 for each temperature. The same held-out test set is used; all generated outputs and corpus scores are saved. This addresses the additional README checklist item without mixing sampled scores into the primary comparison.

No temperature sweep has completed in the supplied run. A possible diversity–fidelity trade-off is a hypothesis, not an observed finding. Run the notebook sweep and review factual errors, repetitions and score variability before drawing a conclusion.

## Execution status and next steps

Run the supplied Colab notebook on a GPU after obtaining model access from Hugging Face. LLaMA and Gemma downloads require the account holder to accept their respective terms and authenticate within the notebook. No model-access gate is bypassed. Rerun all models on one runtime for comparable timing. Regenerate the comparison and documents after the remaining runs complete.

Four automated checks passed in the supplied local run: split separation, future-token masking, exact-match metric sanity, and vocabulary restricted to training data. Those checks validate specific properties; they do not validate unavailable models or prove translation quality. The source archive hash, exact splits, model IDs and measured environment are stored with the project.

## Conclusion

The project implements the required model paths and documents available experimental evidence. The small scratch model exhibits low reference overlap and semantic failures. A four-model winner and a completed architecture comparison cannot be reported until all required runs finish. The remaining work is model execution and bilingual review, not replacement of missing scores with estimates.

## Sources and project

Dataset and attribution — https://www.manythings.org/anki/

Dataset snapshot — https://download.pytorch.org/tutorial/data.zip

mT5 model card — https://huggingface.co/google/mt5-small

LLaMA model card — https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Gemma model card — https://huggingface.co/google/gemma-3-1b-it

SacreBLEU evaluation implementation — https://github.com/mjpost/sacrebleu

Project repository — https://github.com/KothariMayaank/Fgai_Assgn_2/tree/main/assignment3

See results/RESULTS.md for the numerical comparison.
