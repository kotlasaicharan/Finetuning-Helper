
%%capture
!pip install nltk rouge-score sacrebleu

import nltk
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
nltk.download('wordnet')


from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def calculate_scores(df , true , pred):
  """Calculates BLEU, METEOR, and ROUGE scores for a dataframe.

  Args:
    df: A Pandas DataFrame containing 'true_text' and 'predicted_text' columns.

  Returns:
    A new DataFrame with BLEU, METEOR, and ROUGE scores appended.
  """

  bleu_scores = []
  meteor_scores = []
  rouge_scores = []
  bleu1_scores = []
  bleu2_scores = []
  bleu3_scores = []
  bleu4_scores = []
  
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

  for _, row in df.iterrows():
    true_text = row[true]
    predicted_text = row[pred]

    # reference = [['this', 'is', 'very, 'small', 'test']]
    # candidate = ['this', 'is', 'a', 'test']
    # sentence_bleu(reference, candidate)
    reference = [true_text.split()]
    candidate = predicted_text.split()
    bleu1 =  sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    # bleu = sentence_bleu( [true_text.split()] , predicted_text.split() )
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)

    # METEOR Score
    # print (nltk.translate.meteor_score.meteor_score(
    # ["this is an apple", "that is an apple"], "an apple on this tree"))
    # try:
    meteor_score = nltk.translate.meteor_score.single_meteor_score(true_text.split() , predicted_text.split() )
    meteor_scores.append(meteor_score)
    # except:
    #   print(32)
    #   meteor_scores.append(0)  # Handle potential errors


    # ROUGE Score
    rouge = scorer.score(true_text, predicted_text)
    rouge_scores.append(rouge)
  
  all_scored = {}
  all_scored['bleu1'] = bleu1_scores
  all_scored['bleu2'] = bleu2_scores
  all_scored['bleu3'] = bleu3_scores
  all_scored['bleu4'] = bleu4_scores
  all_scored['meteor'] = meteor_scores
  all_scored['rouge1'] = [score['rouge1'].fmeasure for score in rouge_scores]
  all_scored['rouge2'] = [score['rouge2'].fmeasure for score in rouge_scores]
  all_scored['rougeL'] = [score['rougeL'].fmeasure for score in rouge_scores]
  
  scores ={}
  for key, value in all_scored.items():
    scores[key] = round(np.array(value).mean(), 4)

  return scores , all_scored


#explamples
# scores , _ = calculate_scores( df, "reference" , "predicted")