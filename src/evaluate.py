import argparse
import rouge
from pprint import pprint
import os
import time
import numpy as np
import json 
import glob
import pandas as pd

def test_rouge(candidates, references, temp_dir='./tmp'):
    assert len(candidates) == len(references), f'{temp_dir}: len cand {len(candidates)} len ref {len(references)}'

    cnt = len(candidates)
    evaluator = rouge.Rouge()

    all_scores = []

    for cand_idx, cand in enumerate(candidates):
        curr_targets = references[cand_idx]
        curr_scores = []
        if type(curr_targets)==list:
            for tgt in curr_targets:
                r = evaluator.get_scores(cand, tgt)
                curr_scores += r
        else:
            tgt = curr_targets
            r = evaluator.get_scores(cand, tgt)
            curr_scores += r
        # Take the max of curr scores
        max_rouge = 0.
        max_idx = 0
        for score_idx, s in enumerate(curr_scores):
            if s['rouge-1']['f'] > max_rouge:
                max_rouge = s['rouge-1']['f']
                max_idx = score_idx
        all_scores.append(curr_scores[max_idx])
    
    # Average across all scores
    avg_scores = {"rouge-1": {
                    "f": [],
                    "p": [],
                    "r":[]
                    },
                "rouge-2": {
                    "f": [],
                    "p": [],
                    "r": []
                    },
                "rouge-l": {
                    "f": [],
                    "p": [],
                    "r": []
                    }
                }
    for score in all_scores:
        for r_type in score.keys():
            for m_type in score[r_type].keys():
                x = score[r_type][m_type]
                avg_scores[r_type][m_type].append(x)

    for r_type in avg_scores.keys():
        for m_type in avg_scores[r_type].keys():
            x = avg_scores[r_type][m_type]
            avg_scores[r_type][m_type] = np.mean(x)

    return avg_scores

def evaluate(datadir, multitarget_path=None):

    # GET FILE PATHS
    candidate_file = glob.glob(os.path.join(datadir, '*.candidate'))
    if len(candidate_file) > 1:
        raise Exception('More than one candiate file found')
    candidate_file = candidate_file[0]
    id_file = glob.glob(os.path.join(datadir, '*.ids'))

    if len(id_file) > 1:
        raise Exception('More than one id file found')
    id_file = id_file[0]

    author_gold_file = glob.glob(os.path.join(datadir, '*.gold'))
    if len(author_gold_file) > 1:
        raise Exception('More than one gold file found')
    author_gold_file = author_gold_file[0]

    cand = list(map(lambda x: x.strip(), open(candidate_file).readlines()))
    ref = list(map(lambda x: x.strip(), open(author_gold_file).readlines()))
    ids = list(map(lambda x: x.strip(), open(id_file).readlines()))
    
    print('Single target results:')
    pprint(test_rouge(cand, ref))

    if multitarget_path:
        multitarget_lines = map(lambda x: json.loads(x.strip()), open(multitarget_path).readlines())
        multitarget_df = pd.DataFrame(multitarget_lines)  
        auth = pd.DataFrame(list(zip(ids, cand)), columns =['paper_id', 'pred']) 
        merged = pd.merge(left=multitarget_df, right=auth, left_on='paper_id', right_on='paper_id')

        assert len(merged) == len(ids)

        cand = []
        ref = []
        for _, row in merged.iterrows():
            cand.append(row['pred'])
            ref.append([r.lower() for r in row['target']])
        
        print('Multitarget target results:')
        pprint(test_rouge(cand, ref))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--multitarget', help='path to multitarget json test file')
    args = parser.parse_args()

    evaluate(args.datadir, multitarget_path=args.multitarget)

    
