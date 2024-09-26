from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

path = 'models/Meta_qt/evaluate_step10_beam5x4_2000/tmp'
src = list()

with open(f'{path}/output.tok.ref', 'r') as file:
    ref = file.readlines()
with open(f'{path}/output.tok.sys', 'r') as file:
    sys = file.readlines()
assert len(ref) == len(sys)
scores = 0
for i in range(len(ref)):
    scores += scorer.score(ref[i], sys[i])['rougeL'].fmeasure
scores = scores / len(ref)

print(round(scores, 4))