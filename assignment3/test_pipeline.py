"""Checks for causal masking, split leakage, metrics, and train-only vocabulary."""
import unittest
import torch
import experiment as e

class PipelineChecks(unittest.TestCase):
    def test_splits_do_not_overlap(self):
        splits=[e.read(e.ROOT/f'data/{n}.json') for n in ['train','validation','test']]
        for i in range(3):
            for j in range(i+1,3):
                for key in ['id','source','reference']:
                    self.assertFalse({r[key].casefold() for r in splits[i]} & {r[key].casefold() for r in splits[j]})
    def test_causal_mask_blocks_future_tokens(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        m=e.make_basic(20,20).eval();s=torch.tensor([[1,4,5,2]])
        a=torch.tensor([[1,6,7,8]]);b=torch.tensor([[1,6,12,13]])
        with torch.no_grad():x=m(s,a);y=m(s,b)
        self.assertTrue(torch.allclose(x[:,:2],y[:,:2],atol=1e-6))
    def test_perfect_translation_metrics(self):
        refs=['This is a sufficiently long test sentence.','Another full sentence tests this exact match.']
        scores=e.score(refs,refs)
        self.assertAlmostEqual(scores['bleu'],100,places=5)
        self.assertAlmostEqual(scores['chrf'],100,places=5)
    def test_training_vocab_excludes_test_only_words(self):
        cp=e.ROOT/'checkpoints/basic.pt'
        if not cp.exists():self.skipTest('Basic checkpoint not yet produced')
        obj=torch.load(cp,map_location='cpu',weights_only=True)
        for key,vkey in [('source','src_vocab'),('reference','tgt_vocab')]:
            trainwords={w for r in e.read(e.ROOT/'data/train.json') for w in e.tokens(r[key])}
            self.assertEqual(set(obj[vkey])-{'<pad>','<bos>','<eos>','<unk>'},trainwords)

if __name__=='__main__':unittest.main()
