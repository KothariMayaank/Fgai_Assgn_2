# Dataset attribution

The English–French pairs originate from the Tatoeba community, distributed by ManyThings.org and preserved in the public PyTorch tutorial archive.

- Tatoeba: https://tatoeba.org/
- ManyThings dataset page and attribution guidance: https://www.manythings.org/anki/
- Exact snapshot: https://download.pytorch.org/tutorial/data.zip
- Tutorial: https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

ManyThings identifies its sentence-pair data as Creative Commons Attribution 2.0 France and credits Tatoeba contributors. The older two-column PyTorch snapshot does not include the sentence IDs, contributor names or third-column attribution included in newer ManyThings exports. This project retains source-level attribution and clearly records this limitation; it does not claim ownership of the sentences. For publication or redistribution requiring contributor-level provenance, use an attributed current export and regenerate all experiments. No new license is imposed on the underlying dataset.

The committed split JSON files are derived by Unicode and whitespace normalization, length filtering, source/target deduplication and seeded subset selection. Their hashes are in `data/manifest.json`. Dataset licenses and model licenses are separate. Model weights are not included in this repository.
