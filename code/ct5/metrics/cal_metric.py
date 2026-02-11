from smooth_bleu import *
# from smooth_bleu import *
import json
# from nlgeval.pycocoevalcap.cider.cider import Cider
# from nlgeval.pycocoevalcap.meteor.meteor import Meteor
# from nlgeval.pycocoevalcap.rouge.rouge import Rouge
# from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nltk.translate.bleu_score import *
import nltk
from nltk.translate.bleu_score import *
from nltk.translate import meteor
from nltk import word_tokenize
from rouge_score import rouge_scorer

def nltk_bleu_s(goldMap, predictionMap):
    bleu4_score = 0.0
    for key in goldMap.keys():
        pre = predictionMap[key][0].replace(" [SEP] ", " ")
        gold = goldMap[key][0].replace(" [SEP] ", " ")

        bleu4_score += sentence_bleu([gold], pre, smoothing_function=SmoothingFunction().method4)
    
    return bleu4_score / len(goldMap.keys()) * 100


def json_to_map(result, key_hyp = "hypothesis", key_ref = "reference", remove_first_word=False):
    """
    Convert jsonl_file to predictionMap and goldMap
    """
    predictionMap = {}
    goldMap = {}

    for id, row in enumerate(result):
        if not remove_first_word:
            predictionMap[id] = [row[key_hyp]]
            goldMap[id] = [row[key_ref]]
        else:
            if len(row[key_hyp].split(" ", 1)) < 2:
                predictionMap[id] = [""]
            else:
                predictionMap[id] = [row[key_hyp].split(" ", 1)[1]]
            goldMap[id] = [row[key_ref].split(" ", 1)[1]]
    
    print("Total: " + str(len(goldMap)))
    return (goldMap, predictionMap)

def calculate_rouge(goldMap, predictionMap):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rougel_score=0.0

    for key in goldMap.keys():
        pre = predictionMap[key][0].replace(" [SEP] ", " ")
        gold = goldMap[key][0].replace(" [SEP] ", " ")

        scores = scorer.score(gold, pre)
        rougel_score += scores['rougeL'].fmeasure
    
    return rougel_score / len(goldMap.keys()) * 100

def nltk_meteor(goldMap, predictionMap):
    meteor_socre = 0.0
    for key in goldMap.keys():
        pre = predictionMap[key][0].replace(" [SEP] ", " ")
        gold = goldMap[key][0].replace(" [SEP] ", " ")

        meteor_socre += meteor(
            [word_tokenize(gold)],
            word_tokenize(pre)
        )
    
    return meteor_socre / len(goldMap.keys()) * 100

def exact_match(goldMap, predictionMap):
    exact_match = 0
    for key in goldMap.keys():
        pre = predictionMap[key][0].replace(" [SEP] ", " ")
        gold = goldMap[key][0].replace(" [SEP] ", " ")
        
        if pre == gold:
            exact_match += 1
    
    return exact_match / len(goldMap.keys()) * 100

def calculate_metric(result, key_hyp = "hypothesis", key_ref = "reference", remove_first_word=False):

    (goldMap, predictionMap) = json_to_map(result, key_hyp, key_ref, remove_first_word)

    print("smooth_bleu: ", bleuFromMaps(goldMap, predictionMap)[0])

    print("nltk bleu: ", nltk_bleu_s(goldMap, predictionMap))

    print("rouge-l: ", calculate_rouge(goldMap, predictionMap))

    print("meteor: ", nltk_meteor(goldMap, predictionMap))

