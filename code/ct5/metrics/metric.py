from metrics.smooth_bleu import *
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
            predictionMap[id] = [row[key_hyp][:128]]
            goldMap[id] = [row[key_ref][:128]]
        else:
            predictionMap[id] = [row[key_hyp].split(" ", 1)[1][:129]]
            goldMap[id] = [row[key_ref].split(" ", 1)[1][:129]]
    
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

def param_accuracy(goldMap, predictionMap):
    """
    计算准确率
    """
    ans = 0.0
    cnt = 0.0

    # 参数相关的准确性
    f = {}
    # 参数个数
    pn = {}

    #
    perfect = {}

    for key in goldMap.keys():
        a = 0.0
        pre = predictionMap[key][0].split(" [SEP] ")
        gold = goldMap[key][0].split(" [SEP] ")

        pre_param = [tag.strip().split()[0] for tag in pre]
        gold_param = set([tag.strip().split()[0] for tag in gold])

        
        cnt += 1
        for pre_p in pre_param:
            if pre_p in gold_param:
                a += 1

        # 根据参数个数分别统计准确率
        f[len(gold_param)] = f.get(len(gold_param), 0) + a / len(gold_param)
        pn[len(gold_param)] = pn.get(len(gold_param), 0) + 1

        # 完美预测全部参数
        perfect[len(gold_param)] = perfect.get(len(gold_param), 0) + (a == len(gold_param))

        ans += a / len(gold_param)
    
    sorted(f.items(), key=lambda x: x[0])
    for key in f.keys():
        print("参数个数为" + str(key) + "的准确率：" + str(f[key] / pn[key] * 100))
    
    for key in perfect.keys():
        print("参数个数为" + str(key) + "的完美预测率：" + str(perfect[key] / pn[key] * 100))
    
    return ans / cnt * 100


def calculate_metric(result, key_hyp = "hypothesis", key_ref = "reference", remove_first_word=False):
    """
    根据predictionMap和goldMap计算smooth_bleu
    """
    (goldMap, predictionMap) = json_to_map(result, key_hyp, key_ref, remove_first_word)

    print("smooth_bleu: ", bleuFromMaps(goldMap, predictionMap)[0])

    print("nltk bleu: ", nltk_bleu_s(goldMap, predictionMap))

    print("rouge-l: ", calculate_rouge(goldMap, predictionMap))

    print("meteor: ", nltk_meteor(goldMap, predictionMap))

    print("Excatly match: ", exact_match(goldMap, predictionMap))

    print("@Param准确性:", param_accuracy(goldMap, predictionMap))


def calculate_metric_single(result, key_hyp = "hypotheses", key_ref="references"):
    """
    将description分割，重新计算bleu
    """
    # todo 两种评判方式：第一种是全部描述一起生成，两个的构造map不一样。
    # todo 需要加一个控制，生成带参数名时，计算metric时控制不计算参数名
    predictionMap = {}
    goldMap = {}

    cnt = 0
    for id, row in enumerate(result):
        reference_list = row[key_ref].split("[SEP]")
        hypothesis_list = row[key_hyp].split("[SEP]")

        # 将 hypothesis_list中空串去掉
        hypothesis_list = [h for h in hypothesis_list if h.strip() != ""]
        for i in range(len(reference_list)):
            ref = reference_list[i].strip()
            hyp = ""
            for h in hypothesis_list:

                if h.strip().split()[0] == ref.split()[0]:
                    hyp = h
                    break
            predictionMap[cnt] = [hyp]
            goldMap[cnt] = [ref]
            cnt += 1
    print("Total: " + str(len(goldMap)))

    print("Calculating smooth_bleu...")
    print("smooth_bleu: ", bleuFromMaps(goldMap, predictionMap)[0])

    print("Calculating nltk bleu...")
    print("nltk bleu: ", nltk_bleu_s(goldMap, predictionMap))

    print("Calculating rouge-l...")
    print("rouge-l: ", calculate_rouge(goldMap, predictionMap))

    print("Calculating meteor...")
    print("meteor: ", nltk_meteor(goldMap, predictionMap))

    print("Excatly match: ", exact_match(goldMap, predictionMap))

    # print("@Param准确性:", param_accuracy(goldMap, predictionMap))
    # 

if __name__ == '__main__':
    
    with open("./output_dir/codeT5-small/0630/single_sign_add_pname/with_param_name/test_results.jsonl", "r") as f:
        result = [json.loads(line) for line in f]
    calculate_metric(result,key_hyp="hypotheses", key_ref="references", remove_first_word=False)
    # calculate_metric_single(result)
    