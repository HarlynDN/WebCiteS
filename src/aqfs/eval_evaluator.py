import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from argparse import ArgumentParser
import logging
import json
import numpy as np
import tabulate
from sklearn.metrics import cohen_kappa_score

from evaluator import Evaluator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--f', type=str, default='../data/aqfs_snippet/test.json',
                    help='The test set of WebCiteS.')

parser.add_argument('--nli_model', type=str, default=None,
                    help='The path to citation scorer model checkpoint.')

parser.add_argument('--claimsplit_model', type=str, default=None,
                    help='The path to claim-split model checkpoint.')

parser.add_argument('--citation_mask', choices=['auto', 'default', 'human'], default='auto',
                    help='If set to `auto`, predict citation_mask for each sentece using the nli model;'
                    'If set to `default`, assign citation_mask=1 to all sentences;'
                    'If set to `human`, assign citation_mask=1 to sentences with human-annotated citations and 0 otherwise.')

parser.add_argument('--nli_batch_size', type=int, default=256,
                    help='batch size for (premise, hypothesis) pairs')

parser.add_argument('--claimsplit_batch_size', type=int, default=512,
                    help='batch size for claim-split sentences')

parser.add_argument('--nli_max_seq_length', type=int, default=512,
                    help='max length for (premise, hypothesis) pairs')

parser.add_argument('--claimsplit_max_source_length', type=int, default=64,
                    help='max length for claim-split sentences')

parser.add_argument('--claimsplit_max_target_length', type=int, default=128,
                    help='max length for claim-split generations')
                    
parser.add_argument('--output_dir', type=str, default=None,
                    help='If provided, will to save the evaluation results in the directory')

parser.add_argument('--device_map', type=str, default='cuda',
                    help='if set to `cuda`, will do single gpu inference; if set to `auto`, will shard model across multiple gpus and do pipeline inference.')

def init_score_dict():
    score_dict={
        "Citation Precision": [],
        "Citation Recall": [],
        "AIS_AutoCite": [],
        "AIS_HumanCite": [],
        "Agreement Rate": None,
        "Cohen's Kappa": None
    }
    return score_dict


def run_evaluation_on_evaluator():
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    # load test file
    with open(args.f, 'r') as f:
        samples = json.load(f)

    # load evaluator
    evaluator = Evaluator(
        claimsplit_model_path=args.claimsplit_model,
        nli_model_path=args.nli_model,
        device_map=args.device_map
    )      

    all_scores = {'evaluator': init_score_dict()}


    # parse summary with claim-split
    if all(['summary_parsing' in sample.keys() for sample in samples]):
        all_summary_parsing = [sample['summary_parsing'] for sample in samples]
    else:
        all_summary_parsing = evaluator._parse_text(
                                text=[sample['summary'] for sample in samples],
                                claimsplit = evaluator.claimsplit_model is not None,
                                max_source_length=args.claimsplit_max_source_length,
                                max_target_length=args.claimsplit_max_target_length,
                                batch_size=args.claimsplit_batch_size,
                                verbose=True
                            )
    for summary_parsing in all_summary_parsing:
        for item in summary_parsing:
            if args.citation_mask == 'default':
                item['citation_mask'] = 1
            elif args.citation_mask == 'human':
                item['citation_mask'] = 1 if item['citations'] != [] else 0
            item['human_citations'] = item['nearest_citations']

    # compute attribution scores
    all_summary_parsing = evaluator._predict_citation(   
                            text_parsing=all_summary_parsing,
                            docs=[sample['docs'] for sample in samples],
                            query=[sample['query'] for sample in samples],
                            predict_citation_mask= (args.citation_mask == 'auto'),
                            max_seq_length=args.nli_max_seq_length,
                            batch_size=args.nli_batch_size,
                            verbose=True
                        )
    all_attribution_scores = [evaluator._compute_attribution_score(
                                text_parsing=parsing, 
                                citation_pred_key='auto_citations',
                                citation_ref_key='human_citations'
                            )
                        for parsing in all_summary_parsing]
    all_scores['evaluator']['Citation Precision'] = [score['citation_precision'] for score in all_attribution_scores]
    all_scores['evaluator']['Citation Recall'] = [score['citation_recall'] for score in all_attribution_scores]
    all_scores['evaluator']['AIS_HumanCite'] = [score['ais'] for score in all_attribution_scores]
    all_scores['evaluator']['AIS_AutoCite'] = [score['acs'] for score in all_attribution_scores]

    # average scores
    for metric_name, metric_scores in all_scores['evaluator'].items():
        if isinstance(metric_scores, list):
            all_scores['evaluator'][metric_name] = round(np.mean(metric_scores)*100, 2)  if metric_scores != [] else 'N/A'

    # compute agreement rate and cohen's kappa scores on whether a sentence should cite a document
    all_citation_predictions = []
    all_citation_labels = []
    for summary_parsing in all_summary_parsing:
        for item in summary_parsing:
            if item['citation_mask'] == 0:
                continue
            for doc_id in range(1,6):
                all_citation_predictions.append(1 if doc_id in item['auto_citations'] else 0)
                all_citation_labels.append(1 if doc_id in item['human_citations'] else 0)
    agg = round(np.mean([pred==label for pred, label in zip(all_citation_predictions, all_citation_labels)])*100, 2)
    kappa = round(cohen_kappa_score(all_citation_predictions, all_citation_labels), 4)
    all_scores['evaluator']['Agreement Rate'] = agg
    all_scores['evaluator']["Cohen's Kappa"] = kappa

    # print results
    for pred_name, scores in all_scores.items():
        logger.info(f'=== {pred_name} ===')
        for metric_name, metric_scores in scores.items():
            logger.info(f'{metric_name}: {metric_scores}')
        print('======')


    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        metrics_output_name = 'metrics.json'
        with open(os.path.join(args.output_dir, metrics_output_name), 'w') as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)
        logger.info(f'Metrics saved at {os.path.join(args.output_dir, metrics_output_name)}')

if __name__ == '__main__':
    run_evaluation_on_evaluator()