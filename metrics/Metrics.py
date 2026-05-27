from metrics.mscoco_rouge import calc_score
from pathlib import Path
from sacrebleu.metrics import BLEU

class RougeMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_seq = 0
        self.rouge_score = 0.0

    def update(self, preds, targets):
        # preds and targets are both lists of strings
        assert len(preds) == len(targets), "Number of predictions and references must match"
        for h, r in zip(preds, targets):
            self.rouge_score += calc_score(hypotheses=[h], references=[r])
            self.n_seq += 1

    def compute(self):
        if self.n_seq == 0:
            raise ValueError("No samples to compute ROUGE score.")
        return (self.rouge_score / self.n_seq) * 100
    
def calc_translation_score(path, mode='dev'):
    bleu4 = BLEU()
    bleu3 = BLEU(max_ngram_order=3)
    bleu2 = BLEU(max_ngram_order=2)
    bleu1 = BLEU(max_ngram_order=1)

    rouge = RougeMetric()
    rouge.reset()

    pred_path = Path(path) / f'{mode}_pres.txt'
    trg_path = Path(path) / f'{mode}_refs.txt'


    with open(pred_path, 'r') as f:
        preds = f.readlines()
    with open(trg_path, 'r') as f:
        refs = f.readlines()
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]
    assert len(preds) == len(refs), "Number of predictions and references must match"


    rouge.update(preds, refs)
    score = rouge.compute()

    bleu4_s = bleu4.corpus_score(preds, [refs]).score
    bleu3_s = bleu3.corpus_score(preds, [refs]).score
    bleu2_s = bleu2.corpus_score(preds, [refs]).score
    bleu1_s = bleu1.corpus_score(preds, [refs]).score

    print('-----------------------------------')
    print(f'{mode.upper()} Set:')
    print(f"Average ROUGE score: {score:.2f}")
    print(f"BLEU-4 score: {bleu4_s}")
    print(f"BLEU-3 score: {bleu3_s}")
    print(f"BLEU-2 score: {bleu2_s}")
    print(f"BLEU-1 score: {bleu1_s}")
  
