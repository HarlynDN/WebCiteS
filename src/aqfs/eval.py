import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from argparse import ArgumentParser
import logging
import json
import numpy as np
from tqdm import tqdm
import re
from sacrebleu.metrics import BLEU

from evaluator import Evaluator, remove_citations


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--pred_file', type=str, default=None,
                    help='The json file containing the generated predictions.')

parser.add_argument('--nli_model_path', type=str, default=None,
                    help='The path to citation scorer model checkpoint for evaluation.')

parser.add_argument('--claimsplit_model_path', type=str, default=None,
                    help='The path to claim-split model checkpoint.')

parser.add_argument('--nli_batch_size', type=int, default=256,
                    help='batch size for (premise, hypothesis) pairs')

parser.add_argument('--claimsplit_batch_size', type=int, default=512,
                    help='batch size for claim-split sentences')

parser.add_argument('--nli_max_seq_length', type=int, default=512,
                    help='maximum length for (premise, hypothesis) pairs')

parser.add_argument('--claimsplit_max_source_length', type=int, default=64,
                    help='max length for claim-split sentences')

parser.add_argument('--claimsplit_max_target_length', type=int, default=128,
                    help='max length for claim-split generations')

parser.add_argument('--citation_mask', choices=['predict', 'default'], default='predict',
                    help='automatically predict citation mask or simply assign `citation_mask` to 1 for all sentences.')

parser.add_argument('--citation_type', choices=['direct', 'nearest'], default='nearest',
                    help='How to find the citations for a sentence. '
                    'direct: only look at citations directly embedded within the sentence'
                    'nearest: if the sentence does not have direct citations, will also look for the nearest set of citations from the following sentences.')
                    
parser.add_argument('--output_dir', type=str, default=None,
                    help='The directory to save the results, will default to the same directory as `pred_file` if not provided.')

parser.add_argument('--device_map', type=str, default='cuda',
                    help='if set to `cuda`, will do single gpu inference; if set to `auto`, will shard model across multiple gpus and do pipeline inference.')


def init_score_dict():
    return {
        'Length': [],
        'Self-BLEU': [],
        'Claim Precision': [],
        'Claim Recall': [],
        'Claim F1': [],
        'Citation Precision': [],
        'Citation Recall': [],
        'Citation F1': [],
        'AIS': [],
        'ACS': []
    }


def run_evaluation():
    """
    Evaluate the performance on the AQFS task using the automatic evaluator
    """
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pred_file)
        logger.info(f'`--output_dir` is not provided. Evaluation outputs will be saved at {args.output_dir}')

    # load generation file
    with open(args.pred_file, 'r') as f:
        gen_samples = json.load(f)

    # load evaluator
    evaluator = Evaluator(
        nli_model_path=args.nli_model_path,
        claimsplit_model_path=args.claimsplit_model_path,
        device_map=args.device_map
    )

    # collect predictions
    sample = gen_samples[0] # check number of generations per input
    # if only one generation per input
    if isinstance(sample['predict'], str): 
        pred_names = ['predict']
        all_predictions = [{'predict': _sample['predict']} for _sample in gen_samples]

    # if a list of generations per input,evaluate all of them
    elif isinstance(sample['predict'], list): 
        pred_names = [f'predict_{i}' for i in range(len(sample['predict']))]
        all_predictions = [{name: _sample['predict'][i] for i, name in enumerate(pred_names)} for _sample in gen_samples]

    # if a dict of generations per input, evaluate all of them
    elif isinstance(sample['predict'], dict): 
        pred_names = [name for name, pred in sample['predict'].items() if pred is not None]
        if isinstance(sample['predict'][pred_names[0]], str):
            all_predictions = [_sample['predict'] for _sample in gen_samples]
        else: 
            all_predictions = [{k: v['text'] for k, v in _sample['predict'].items() if k in pred_names} for _sample in gen_samples]
    else:
        raise ValueError(f'Invalid prediction format: {type(sample["predict"])}')
    
    all_scores = {pred_name: init_score_dict() for pred_name in pred_names}

    # compute all metrics
    # length:
    for predictions in all_predictions:
        for pred_name, pred in predictions.items():
            all_scores[pred_name]['Length'].append(len(remove_citations(pred)))

    # Self-BLEU
    bleu = BLEU(tokenize='zh', effective_order=True)

    def self_bleu(text: str):
        sens = [sen for sen in re.split('(?<=[。；！？;!?\n])', text) if sen] # segment sentences
        if len(sens) <= 1:
            return 0.
        scores = []
        for i in range(len(sens)):
            scores.append(bleu.sentence_score(sens[i], sens[:i] + sens[i+1:]).precisions[-1])
        return np.mean(scores)
    
    for sample, predictions in tqdm(zip(gen_samples, all_predictions), total=len(gen_samples), desc='computing Self-BLEU'):
        for pred_name, pred in predictions.items():
            text = remove_citations(pred)
            all_scores[pred_name]['Self-BLEU'].append(self_bleu(text))

    # claim scores
    all_preds, all_refs = [], []
    for sample, predictions in zip(gen_samples, all_predictions):
        ref = sample['label']
        for pred_name, pred in predictions.items():
            all_refs.append(ref)
            all_preds.append(pred)
    logger.info(f'Computing ClaimScore')
    claimscores = evaluator.evaluate_summary(
                                predictions=all_preds,
                                references=all_refs,
                                claimsplit_max_source_length=args.claimsplit_max_source_length,
                                claimsplit_max_target_length=args.claimsplit_max_target_length,
                                claimsplit_batch_size=args.claimsplit_batch_size,
                                nli_max_seq_length=args.nli_max_seq_length,
                                nli_batch_size=args.nli_batch_size,
                                verbose=True
                            )
    claim_p, claim_r = claimscores['claim_precision'], claimscores['claim_recall']
    # reshape to (num_samples, num_pred_names)
    claim_p = np.array(claim_p).reshape((len(all_predictions), len(pred_names)))
    claim_r = np.array(claim_r).reshape((len(all_predictions), len(pred_names)))

    for i in range(len(all_predictions)):
        for j, pred_name in enumerate(pred_names):
            all_scores[pred_name]['Claim Precision'].append(claim_p[i, j])
            all_scores[pred_name]['Claim Recall'].append(claim_r[i, j])

    # attribution scores   
    all_summaries, all_docs, all_queries = [], [], []
    for sample, predictions in zip(gen_samples, all_predictions):
        all_summaries.extend([pred for pred in predictions.values()])
        all_docs.extend([sample['docs'] for _ in range(len(predictions))])
        all_queries.extend([sample['query']]*len(predictions))
    logger.info(f'Computing citation scores')
    all_attribution_scores, all_summary_parsings = evaluator.evaluate_attribution(
                                                    summary = all_summaries, 
                                                    docs = all_docs, 
                                                    query = all_queries,
                                                    predict_citation_mask=(args.citation_mask=='predict'),
                                                    citation_pred_key='nearest_citations' if args.citation_type=='nearest' else 'citations',
                                                    claimsplit_max_source_length=args.claimsplit_max_source_length,
                                                    claimsplit_max_target_length=args.claimsplit_max_target_length,
                                                    claimsplit_batch_size=args.claimsplit_batch_size,
                                                    nli_max_seq_length=args.nli_max_seq_length,
                                                    nli_batch_size=args.nli_batch_size,
                                                    verbose=True
                                                )
    all_summary_parsings = [{name: all_summary_parsings[i*len(pred_names)+j] for j, name in enumerate(pred_names)} for i in range(len(gen_samples))]
    for metric_name in all_attribution_scores.keys():
        all_attribution_scores[metric_name] = np.array(all_attribution_scores[metric_name]).reshape((len(gen_samples), len(pred_names)))
    for i in range(len(gen_samples)):
        for j, pred_name in enumerate(pred_names):
            all_scores[pred_name]['Citation Precision'].append(all_attribution_scores['citation_precision'][i, j])
            all_scores[pred_name]['Citation Recall'].append(all_attribution_scores['citation_recall'][i, j])
            all_scores[pred_name]['AIS'].append(all_attribution_scores['ais'][i, j])
            all_scores[pred_name]['ACS'].append(all_attribution_scores['acs'][i, j])
    
    # aggregate evaluation results
    eval_results = []
    for i, prediction in enumerate(all_predictions):
        item = {'query': gen_samples[i]['query']}
        for pred_name, pred in prediction.items():
            item[pred_name] = {
                'text': pred,
                'parsing': all_summary_parsings[i][pred_name],
                'scores': {metric_name: round(scores[i], 4) for metric_name, scores in all_scores[pred_name].items() if scores}
            }
        eval_results.append(item)


    # compute average scores and compute F1
    for pred_name, scores in all_scores.items():
        for metric_name, metric_scores in scores.items():
            if metric_scores != []:
                all_scores[pred_name][metric_name] = np.mean(metric_scores)
                if metric_name not in ['Length', 'Self-BLEU']:
                    all_scores[pred_name][metric_name] = round(all_scores[pred_name][metric_name]*100, 2)
        claim_p, claim_r = all_scores[pred_name]['Claim Precision'], all_scores[pred_name]['Claim Recall']
        all_scores[pred_name]['Claim F1'] = round(2*claim_p*claim_r/(claim_p+claim_r), 2) if claim_p+claim_r != 0 else 0
        citation_p, citation_r = all_scores[pred_name]['Citation Precision'], all_scores[pred_name]['Citation Recall']
        all_scores[pred_name]['Citation F1'] = round(2*citation_p*citation_r/(citation_p+citation_r), 2) if citation_p+citation_r !=0 else 0

    # print results
    for pred_name, scores in all_scores.items():
        logger.info(f'=== {pred_name} ===')
        for metric_name, metric_scores in scores.items():
            logger.info(f'{metric_name}: {metric_scores}')
        print('======')


    metrics_output_name = 'metrics_' + os.path.basename(args.pred_file).split('.')[0] + '.json'
    with open(os.path.join(args.output_dir, metrics_output_name), 'w') as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    logger.info(f'Metrics saved at {os.path.join(args.output_dir, metrics_output_name)}')

    eval_results_output_name = 'eval_results_' + os.path.basename(args.pred_file).split('.')[0]+'.json'
    with open(os.path.join(args.output_dir, eval_results_output_name), 'w') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions saved at {os.path.join(args.output_dir, eval_results_output_name)}")

if __name__ == '__main__':
    run_evaluation()