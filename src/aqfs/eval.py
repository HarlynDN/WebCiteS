import os
from argparse import ArgumentParser
import logging
import json
from typing import List
import numpy as np
from tqdm import tqdm
import re
from sacrebleu.metrics import BLEU
from nltk import word_tokenize, sent_tokenize

from evaluator import Evaluator, remove_citations, is_chinese


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = ArgumentParser()
parser.add_argument('--f', type=str, default=None,
                    help='The json file containing the generations to evaluate.')

parser.add_argument('--max_eval_samples', type=int, default=None,
                    help='Could be used to limit the number of samples to evaluate.')

parser.add_argument('--nli_model', type=str, default=None,
                    help='The path to citation scorer model checkpoint for evaluation.')

parser.add_argument('--claimsplit_model', type=str, default=None,
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
                    help='The directory to save the results, will default to the same directory as `f` if not provided.')

parser.add_argument('--device', type=str, default='cuda',
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
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.f)
        logger.info(f'`--output_dir` is not provided. Evaluation outputs will be saved at {args.output_dir}')

    # load model output file
    eval_file = json.load(open(args.f, 'r'))
    if isinstance(eval_file, list):
        eval_data, exp_args = eval_file, {}
    else:
        exp_args = eval_file['args'] if 'args' in eval_file else {}
        eval_data = eval_file['data']

    if args.max_eval_samples:
        eval_data = eval_data[:args.max_eval_samples]
    
    query_key = 'query' if 'query' in eval_data[0].keys() else 'question' if 'question' in eval_data[0].keys() else None
    pred_key = 'predict' if 'predict' in eval_data[0].keys() else 'output'
    label_key = 'label' if 'label' in eval_data[0].keys() else 'answer'
    for i, sample in enumerate(eval_data):
        if isinstance(sample[pred_key], list):
            sample[pred_key] = sample[pred_key][0]
            if i == 0:
                logger.warning(f'Only the first prediction per sample is used for evaluation.')

    all_scores = init_score_dict()

    # load evaluator
    evaluator = Evaluator(
        nli_model_path=args.nli_model,
        claimsplit_model_path=args.claimsplit_model,
        device_map=args.device
    )


    # compute all metrics
    # length:
    for sample in eval_data:
        pred = sample[pred_key]
        not_chinese = re.search("[\u4e00-\u9FFF]", pred) is None
        if not_chinese:
            all_scores['Length'].append(len(word_tokenize(remove_citations(pred))))
        else: # count number of characters for chinese
            all_scores['Length'].append(len(remove_citations(pred)))

    # Self-BLEU
    def self_bleu(text: str):
        if is_chinese(text):
            bleu = BLEU(tokenize='zh', effective_order=True)
            sens = [sen for sen in re.split('(?<=[。；！？;!?\n])', text) if sen]
        else:
            bleu = BLEU(effective_order=True)
            sens = sent_tokenize(text)
        if len(sens) <= 1:
            return 0.
        scores = []
        for i in range(len(sens)):
            scores.append(bleu.sentence_score(sens[i], sens[:i] + sens[i+1:]).precisions[-1])
        return np.mean(scores)
    
    for sample in tqdm(eval_data, total=len(eval_data), desc='computing Self-BLEU'):
        pred = remove_citations(sample[pred_key])
        all_scores['Self-BLEU'].append(self_bleu(pred))

    # claim scores
    logger.info(f'Computing claim scores')
    claimscores = evaluator.evaluate_claims(
                                predictions=[remove_citations(sample[pred_key]) for sample in eval_data],
                                references=[remove_citations(sample[label_key]) for sample in eval_data],
                                claimsplit_max_source_length=args.claimsplit_max_source_length,
                                claimsplit_max_target_length=args.claimsplit_max_target_length,
                                claimsplit_batch_size=args.claimsplit_batch_size,
                                nli_max_seq_length=args.nli_max_seq_length,
                                nli_batch_size=args.nli_batch_size,
                                verbose=True
                            )
    all_scores['Claim Precision'] = claimscores['claim_precision']
    all_scores['Claim Recall'] = claimscores['claim_recall']


    def _format_docs(docs: List):
        """
        template: [{ID}](Title: {T}){P}
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                title, text = doc['title'], doc['text']
            else: # assume separated by \n
                title, text = doc.split("\n", 1)
            formatted_docs.append(f'[{i+1}](Title: {title}){text}')
        return formatted_docs

    logger.info(f'Computing attribution scores')
    all_attribution_scores, all_text_parsings = evaluator.evaluate_attribution(
                                                    generation = [sample[pred_key] for sample in eval_data], 
                                                    docs = [_format_docs(sample['docs']) for sample in eval_data], 
                                                    query = [sample[query_key] for sample in eval_data] if query_key else None,
                                                    predict_citation_mask=(args.citation_mask=='predict'),
                                                    citation_pred_key='nearest_citations' if args.citation_type=='nearest' else 'citations',
                                                    claimsplit_max_source_length=args.claimsplit_max_source_length,
                                                    claimsplit_max_target_length=args.claimsplit_max_target_length,
                                                    claimsplit_batch_size=args.claimsplit_batch_size,
                                                    nli_max_seq_length=args.nli_max_seq_length,
                                                    nli_batch_size=args.nli_batch_size,
                                                    verbose=True
                                                )
    all_scores['Citation Precision'] = all_attribution_scores['citation_precision']
    all_scores['Citation Recall'] = all_attribution_scores['citation_recall']
    all_scores['AIS'] = all_attribution_scores['ais']
    all_scores['ACS'] = all_attribution_scores['acs']

    for i, item in enumerate(eval_data):
        item['parsing'] = all_text_parsings[i]
        item['scores'] = {metric_name: round(scores[i], 4) for metric_name, scores in all_scores.items() if scores}



    # compute average scores and compute F1
    for metric_name, metric_scores in all_scores.items():
        if metric_scores != []:
            all_scores[metric_name] = np.mean(metric_scores)
            if metric_name not in ['Length', 'Self-BLEU']:
                all_scores[metric_name] = round(all_scores[metric_name]*100, 2)
    claim_p, claim_r = all_scores['Claim Precision'], all_scores['Claim Recall']
    all_scores['Claim F1'] = round(2*claim_p*claim_r/(claim_p+claim_r), 2) if claim_p+claim_r != 0 else 0
    citation_p, citation_r = all_scores['Citation Precision'], all_scores['Citation Recall']
    all_scores['Citation F1'] = round(2*citation_p*citation_r/(citation_p+citation_r), 2) if citation_p+citation_r !=0 else 0

    # print results
    logger.info(f"=== {args.f.split('/')[-1]} ===")
    for metric_name, score in all_scores.items():
        logger.info(f'{metric_name}: {score}')
    print('======')


    eval_results = {
        'exp_args': exp_args,
        'eval': {
            'scores': all_scores,
            'eval_args': args.__dict__,
        },
        'data': eval_data
    }
    fname = os.path.basename(args.f).strip('.json')

    json.dump(eval_results, open(os.path.join(args.output_dir, f'eval_{fname}.json'), 'w'), indent=4, ensure_ascii=False)

if __name__ == '__main__':
    run_evaluation()