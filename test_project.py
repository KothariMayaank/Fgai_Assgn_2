"""Lightweight correctness checks: python -m unittest -v test_project.py."""
import unittest
import torch
from train import LanguageModel, MODELS, prepare, tokenize, generate

class ProjectTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("Alice's tea, CAN’T wait!"),["alice's",'tea',"can't",'wait'])

    def test_sequence_alignment(self):
        splits,vocab,stats=prepare(12)
        self.assertEqual(stats['total_words'],sum(stats[s]['words'] for s in ['train','validation','test']))
        for (x,y),name in zip([d.tensors for d in splits],['train','validation','test']):
            self.assertEqual(len(y),stats[name]['words']-12)
            self.assertTrue(torch.equal(x[1:,-1],y[:-1]))
            self.assertTrue(torch.equal(x[1:,:-1],x[:-1,1:]))

    def test_models_and_gradients(self):
        for kind in MODELS:
            model=LanguageModel(kind,20)
            logits=model(torch.randint(20,(3,12)))
            self.assertEqual(tuple(logits.shape),(3,20))
            logits.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sampling_reproducibility(self):
        model=LanguageModel('GRU',4)
        a=generate(model,['<unk>','alice','was','here'],'alice',1.,12,length=5)
        b=generate(model,['<unk>','alice','was','here'],'alice',1.,12,length=5)
        self.assertEqual(a,b)
        self.assertNotIn('<unk>',a['text'])
        with self.assertRaises(ValueError):
            generate(model,['<unk>','alice','was','here'],'alice',0,12)

if __name__=='__main__':
    unittest.main()
