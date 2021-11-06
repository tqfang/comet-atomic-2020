import json
import pandas as pd
import numpy as np
from nlgeval.pycocoevalcap.bleu.bleu import Bleu

def read_jsonl_lines(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def get_gen(strs):
    strs = strs.split()
    st = 0
    ed = 0
    for i in range(len(strs)):
        if strs[i] == "[GEN]":
            st = i
        if strs[i] == "[EOS]":
            ed = i
            break
    return " ".join(strs[st+1:ed])


def get_reference_sentences(filename):
    result = []
    with open(filename) as file:
        for line in file:
            result.append([x.strip() for x in line.split('\t')[1].split('|')])
    return result

def get_heads_and_relations(filename):
    result = []
    with open(filename) as file:
        for line in file:
            line = line.split('\t')[0]
            head_event = line.split('@@')[0].strip()
            relation = line.split('@@')[1].strip()
            to_add = {
                'head': head_event,
                'relation': relation
            }
            result.append(to_add)
    return result


def main():
    pred = read_jsonl_lines("../models/comet_gpt2b_inversedPopAtomic2020/test_for_gpt2.tsv_pred_generations.jsonl")
    
    selected_ground_tails = get_reference_sentences("../system_eval/test.tsv")
    selected_generations = [get_gen(pred[i]['generations'][0]) for i in range(len(selected_ground_tails))]

    hyps = {idx: [strippedlines] for (idx, strippedlines) in enumerate(selected_generations)}
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(selected_ground_tails)}
    print(" | ".join(
        [str(round(score, 3)) for score in Bleu(4).compute_score(refs, hyps)[0]]
        )
    )

if __name__ == '__main__':
    main()
    '0.408 | 0.231 | 0.153 | 0.107'
