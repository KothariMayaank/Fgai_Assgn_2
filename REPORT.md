# Experimental results and analysis

## Dataset and experimental conditions

The processed corpus contains 26,683 words and 2,628 distinct observed word types. The training-only vocabulary contains 1,293 tokens including `<unk>`. Training, validation and test contain 21,346, 2,668 and 2,669 words respectively, producing 21,334, 2,656 and 2,657 input-target windows within their respective partitions. The unknown-word fractions are 4.84%, 11.73% and 12.85%. The relatively large test unknown fraction means perplexity describes prediction over a reduced vocabulary, not exact prediction of every original word.

All models were trained on CPU for 20 epochs with seed 42, context 12, batch size 128, embedding 64, hidden size 96 per direction, Adam learning rate 0.003 and gradient clipping at 1.0. Each best checkpoint was selected by validation loss. Full environment and epoch histories are supplied as machine-readable files.

## Quantitative comparison

| Model | Parameters | Selected epoch | Validation loss | Test loss | Test perplexity | Test accuracy | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| RNN | 223,725 | 5 | 4.5740 | 4.7110 | 111.16 | 17.39% | 25.28 |
| BiRNN | 363,405 | 2 | 4.7009 | 4.8193 | 123.87 | 14.72% | 39.50 |
| LSTM | 270,381 | 5 | 4.4912 | 4.5836 | 97.86 | 17.95% | 35.99 |
| GRU | 254,829 | 4 | 4.5009 | 4.6191 | 101.40 | 18.18% | 44.27 |

The smoothed unigram baseline achieved test loss 5.3221, perplexity 204.81 and accuracy 8.47%. This baseline predicts from training target frequencies without context.

Training time measures the 20 training passes only, excluding evaluation, serialization and generation. Parameter counts differ: BiRNN doubles the directional state and classifier input, while gated networks have additional recurrent weights. Timing is specific to this run and should not be treated as a hardware-independent speed ranking.

![Validation trajectories and held-out perplexity](results/comparison.png)

## Temperature and diversity

Each row averages nine 50-word continuations: three shared prompts times three random draws. Only generated words are scored. Repetition is one minus distinct-2; these columns are complementary rather than independent measures.

| Model | Temperature | Distinct-1 | Distinct-2 | Repeated bigram fraction |
|---|---:|---:|---:|---:|
| RNN | 0.5 | 0.649 | 0.902 | 0.098 |
| RNN | 1.0 | 0.824 | 0.993 | 0.007 |
| RNN | 1.5 | 0.918 | 1.000 | 0.000 |
| BiRNN | 0.5 | 0.578 | 0.887 | 0.113 |
| BiRNN | 1.0 | 0.822 | 0.995 | 0.005 |
| BiRNN | 1.5 | 0.938 | 1.000 | 0.000 |
| LSTM | 0.5 | 0.653 | 0.927 | 0.073 |
| LSTM | 1.0 | 0.820 | 0.991 | 0.009 |
| LSTM | 1.5 | 0.918 | 0.998 | 0.002 |
| GRU | 0.5 | 0.589 | 0.859 | 0.141 |
| GRU | 1.0 | 0.764 | 0.977 | 0.023 |
| GRU | 1.5 | 0.918 | 1.000 | 0.000 |

![Temperature versus bigram diversity](results/temperature.png)

Low temperature sharpens the output distribution, favoring common phrases. Higher temperature flattens it and increases the chance of less frequent words. Diversity can therefore improve while grammatical or contextual consistency declines. The samples below show why diversity alone cannot identify the best generator.

## Representative generated passages

These are complete 50-word continuations plus the shared seed, using replicate 1. They are selected by fixed seed and replicate, not by fluency. All 108 outputs, including the other two prompts, are in `results/samples.csv`.

### RNN at temperature 0.5

> alice was beginning to get very tired of it was very glad to find it and very uncomfortable to me said alice as she could not could not we said to herself and she thought it would be getting to be ashamed of the table of hearts and she tried to the cheshire cat said alice as

### RNN at temperature 1.0

> alice was beginning to get very tired of it was just now to they all round very good tone and then a little majesty she walked my between filled of things they are said alice growing and writing shoulders am off in a few voice of things mad she very eel but she painting keep queer it

### RNN at temperature 1.5

> alice was beginning to get very tired always i follow hold said its they had gone through lefthand tone and either a book majesty directly back to happen filled of isn't they marked alice's fancy that them cause shoulders am off in his idea oh of things mad she very eel cook thought like they're off knocking

### BiRNN at temperature 0.5

> alice was beginning to get very tired of it was a very good opportunity of the footman was a little thing a little startled she was her and she could not don't know said alice and she could not like it was a few of of things and she felt to her it and was over and

### BiRNN at temperature 1.0

> alice was beginning to get very tired always and the lefthand hold of they to see very lefthand tone and then a little majesty be the soldiers and filled of isn't they marked be face it them to grow back to in a few voice of things mad and very eel cook all round alice found it

### BiRNN at temperature 1.5

> alice was beginning to get very tired always came follow just said its they question see very lefthand tone underneath pig a time majesty be waistcoat my useful filled of isn't never marked on face tired them to shoulders queen can somebody to smiled not of things mad what's very eel cook thought like keep queer knocking

### LSTM at temperature 0.5

> alice was beginning to get very tired of it and she said to herself to see it was a little startled she looked up to her head and she began to get for the king and the other and the duchess in a few minutes it had never heard to be a little thing she found it

### LSTM at temperature 1.0

> alice was beginning to get very tired of room and had made its they question away yer a tone and then he looked but she walked about the dormouse of knot and marked out a low voice to shoulders you can draw well would not of things mad she very called but she found poison off it

### LSTM at temperature 1.5

> alice was beginning to get very tired always handed them hold i'm its they question away yer lefthand next underneath either a time majesty does waistcoat my useful filled of isn't never marked on fancy that them cause shoulders queen off somebody down would not of things mad what's very eel eye thought above keep queer as

### GRU at temperature 0.5

> alice was beginning to get very tired of the door and the moral of the gryphon and the next of the great time she was to her own how she went on in a moment that she found herself in the house and the little of things and she very called but she could not remember it

### GRU at temperature 1.0

> alice was beginning to get very tired of showing which was very good opportunity to see it was next underneath either and looked majesty the waistcoat came and filled to isn't they all said alice and the cause and the game was now would not of things mad she very called but she painting often larger it

### GRU at temperature 1.5

> alice was beginning to get very tired always came follow knowing said its they think see yer lefthand next me either a time majesty does to my useful filled of isn't they marked on face so them to shoulders am off in his idea oh of things mad what's very eel cook thought there keep queer as

## Qualitative interpretation

The following observations are an informal reading of the representative passages above, not numerical human ratings. Coherence means readable local phrasing; contextual consistency means maintaining the seed's subject and situation rather than merely reusing character names.

| Model | Coherence and contextual consistency | Diversity and repetition |
|---|---|---|
| RNN | At 0.5, phrases such as `said alice` and `she thought it would be` resemble the source style, but the output shifts between the table, hearts and the Cheshire Cat without a clear event. At 1.5, `always i follow hold` is already syntactically broken. | The 0.5 passage repeats `could not`; average repeated-bigram fraction is 9.75%. Distinct-2 rises to 1.000 at 1.5 despite poor readability. |
| BiRNN | At 0.5, `a very good opportunity` is a plausible local phrase, followed by inconsistent clauses such as `she could not don't know`. It does not sustain Alice's situation. Extra directional capacity does not improve held-out quality here. | Repeated-bigram fraction is 11.34% at 0.5. Temperature 1.5 yields the highest distinct-1 (0.938), but disconnected word choices dominate the displayed sample. |
| LSTM | At 0.5, `she said to herself`, `she looked up` and `in a few minutes` give comparatively readable short stretches. The transition among the king, duchess and Alice remains weak, and `she began to get for the king` is ungrammatical. High-temperature output also loses syntax. | It has the lowest repeated-bigram fraction at 0.5 (7.26%) and the highest distinct-2 at that temperature (0.927). This supports its local readability advantage without proving long-range coherence. |
| GRU | At 0.5, `she found herself in the house` and `she could not remember it` are plausible fragments. References to the door, Gryphon and house are loosely connected; they do not form a consistent story. | At 0.5 it has the largest repeated-bigram fraction (14.06%). Increasing temperature raises diversity but the representative 1.5 passage becomes a disconnected word sequence. |

All four networks retain words and phrases from the book, but none reliably preserves a narrative across 50 generated words. Increasing temperature from 0.5 to 1.5 reduces bigram repetition for every model and increases lexical diversity. The high-temperature examples show that near-perfect distinct-2 can simply reflect incoherent combinations.

LSTM has the lowest validation and test loss; GRU has the highest test top-1 accuracy (18.18% versus LSTM's 17.95%). These metrics need not agree: accuracy evaluates only the most probable word, whereas cross entropy evaluates the probability assigned to the true word. The small accuracy gap is not evidence of a robust architecture-level difference without repeated runs.

Every model overfits during the fixed 20-epoch budget. Best validation checkpoints occur at epochs 5, 2, 5 and 4 for RNN, BiRNN, LSTM and GRU respectively. BiRNN's training loss falls to 0.4440 while its final validation loss rises to 7.4681, a particularly strong example. Selecting checkpoints by validation loss avoids reporting those overfitted final-epoch weights as the main results. All four selected models outperform the unigram baseline, indicating that learned context contributes useful predictive information.

## Conclusion

The recommended model for this experimental configuration is **LSTM**, chosen by the lowest validation cross entropy (4.4912). Its held-out test perplexity is 97.86, and next-word accuracy is 17.95%. This is an empirical choice for this dataset and training budget, not a claim that the architecture is universally best.

For demonstration text, temperature 0.5 is the conservative choice among the tested settings when local readability matters most; temperature 1.0 offers more varied output, and 1.5 is useful for illustrating how sampling diversity can undermine coherence. These recommendations are qualitative, not based on a blinded human preference experiment.

The small vocabulary, absence of punctuation, 12-word context and single training seed limit the strength of any conclusion. All models can reproduce corpus-style phrases while failing to maintain a coherent narrative. Improvements should be evaluated on fresh validation/test partitions or repeated seeds rather than chosen by inspecting the current test results.
