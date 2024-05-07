import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle

def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    if m < n:
        s, t = t, s
        m, n = n, m
    d = [list(range(n + 1))] + [[i] + [0] * n for i in range(1, m + 1)]
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]
 
def compute_similarity(input_string, reference_string):
    distance = levenshtein_distance(input_string, reference_string)
    max_length = max(len(input_string), len(reference_string))
    similarity = 1 - (distance / max_length)
    return similarity

def validate(GT, PRED, args, OUTPUT_FILE=None):
    if args.dataset_name == 'avqa':
        ACC = {'Temporal':0.0,'Localis':0.0,'Existential':0.0, 'Common Sense/World Knowledge':0.0} 
        COUNT = {'Temporal':0,'Localis':0,'Existential':0, 'Common Sense/World Knowledge':0}
    elif args.dataset_name in ['music_avqa', 'avqa']:
        ACC = {'Counting':0.0,'Temporal':0.0,'Location':0.0,'Comparative':0.0,'Existential':0.0}
        COUNT = {'Counting':0,'Temporal':0,'Location':0,'Comparative':0,'Existential':0}
    elif args.dataset_name == 'audioset': ## TO BE UPDATED
        ACC = {'Before Next':0.0,'Come From':0.0,'Happening':0.0,'Used For':0.0,'When':0.0, 'Where':0.0, 'Which':0.0, 'Why':0.0, 'What':0.0}
        COUNT = {'Before Next':0,'Come From':0,'Happening':0,'Used For':0,'When':0, 'Where':0, 'Which':0, 'Why':0, 'What':0}
    elif args.dataset_name == 'compa':
        ACC = {'compositional_attribute': 0.0}
        COUNT = {'compositional_attribute': 0.0}
    else:
        raise NameError('Invalid dataset name!!!')
    
    
    # ACC = {'which':0.0,'source':0.0,'happen':0.0,'where':0.0,'why':0.0, 'appears':0.0, 'when':0.0, 'doing':0.0, 
    #        'how many':0.0, 'shown':0.0}
    # COUNT = {'which':0,'source':0,'happen':0,'where':0,'why':0, 'appears':0, 'when':0, 'doing':0, 
    #        'how many':0, 'shown':0}
    # ACC = {'Before Next':0.0,'Come From':0.0,'Happening':0.0,'Used For':0.0,'When':0.0, 'Where':0.0, 'Which':0.0, 'Why':0.0, 'What':0.0}
    # COUNT = {'Before Next':0,'Come From':0,'Happening':0,'Used For':0,'When':0, 'Where':0, 'Which':0, 'Why':0, 'What':0}
    
    
    
    
    print('validating...')

    formatted_pred = {}
    for pred in PRED:
        formatted_pred[pred["id"]] = pred

    for gt in GT:
        if 'val' in gt['data_split']:
            id = gt["id"]
            question = gt["question"].lower()
            question_type = gt["question_type"].lower()
            answer = gt["answer"]
            correct_class = gt["correct_class"]
            
            try:
                if args.dataset_name == 'compa':
                    pred_ans = formatted_pred[id]['extracted_ans'].split(") ")[0]
                    pred_ans = pred_ans.replace('(', '').lower()
                    # print('=========> pred_ans', pred_ans)
                else:
                    pred_ans = formatted_pred[id]['extracted_ans'].split(") ")[-1]
                    pred_ans = pred_ans.replace('.', '').lower()

                for k in ACC.keys():
                    if k.lower() in question_type:
                    # if k in question:
                        if answer == "None of the above":
                            if pred_ans.lower() in ["don't know", "do not know", "can't answer", "cannot answer", "none", "no", "no answer", "none of the above"]:
                                ACC[k] += 1.0
                            if args.task_name in ['AAD', 'IASD']:
                                if compute_similarity(correct_class, pred_ans) >= 0.75 or pred_ans in correct_class:
                                    ACC[k] += 1.0
                            COUNT[k] += 1
                        elif args.dataset_name == 'compa':
                            if pred_ans == 'a':
                                ACC[k] += 1.0
                            COUNT[k] += 1
                        else:
                            if compute_similarity(answer, pred_ans) >= 0.5 or pred_ans in answer:
                                ACC[k] += 1.0
                            COUNT[k] += 1
                
            except:
                pass
    
    total_acc = 0.0
    total_count = 0
    for k in ACC.keys():
        total_acc += ACC[k]
        total_count += COUNT[k]
    
    total_acc = total_acc / total_count

    # for k in ACC.keys():
    #     ACC[k] = ACC[k] / COUNT[k]

    metrics = {'TOTAL': {'ACC': total_acc, 'COUNT': total_count},
               'INDIVIDUAL': [ACC, COUNT]}

    print('Total Acc: ', str(total_acc))
    
    if OUTPUT_FILE is not None:
        f = open(OUTPUT_FILE, 'w')
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', default='instruct_test_Absent_Answer_Detection.json', type=str)
    parser.add_argument('--pred_file', default='Absent_Answer_Detection_test_f96_result.json', type=str)
    parser.add_argument('--output_file', default='metrics_result.json', type=str)
    parser.add_argument('--dataset_name', default='avqa', type=str)
    parser.add_argument('--task_name', default='AAD', type=str)
    args = parser.parse_args()
    
    f = open(args.gt_file, 'r')
    gt_data = json.load(f)

    f = open(args.pred_file, 'r')
    pred_data = json.load(f)

    validate(gt_data, pred_data, args, OUTPUT_FILE=args.output_file)

# RUNNING COMMAND
# python metrics.py --gt_file instruct_test_Absent_Answer_Detection.json \
# --pred_file Absent_Answer_Detection_test_f96_result.json \
# --output_file metrics_result.json

# python metrics.py --gt_file /home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/gt/aad/instruct_test_Absent_Answer_Detection.json --pred_file /home/mila/s/subhrajyoti.dasgupta/scratch/videollama/results/Absent_Answer_Detection/Absent_Answer_Detection_test_f96_result_nota.json --output_file /home/mila/s/subhrajyoti.dasgupta/scratch/videollama/scores/metrics_result_nota.json
