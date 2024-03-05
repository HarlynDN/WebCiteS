import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from argparse import ArgumentParser
import logging
import json
import tabulate

from evaluator import Evaluator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--pred_file', type=str, default=None,
                    help='The path of generated predictions.')

parser.add_argument('--nli_model_path', type=str, default="../../../LLMs/mt5-large-finetuned-mnli-xtreme-xnli",
                    help='The path to the nli model for evaluation.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size for (premise, hypothesis) pairs')

parser.add_argument('--max_seq_length', type=int, default=512,
                    help='maximum length for (premise, hypothesis) pairs')

parser.add_argument('--output_dir', type=str, default=None,
                    help='The directory to save the results, default to the directory of `pred_file`.')

parser.add_argument('--device_map', type=str, default='cuda',
                    help='if set to `cuda`, will do single gpu inference; if set to `auto`, will shard model across multiple gpus and do pipeline inference.')



def run_evaluation():
    """
    Evaluate the performance of the claim-split model.
    """
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pred_file)
        logger.info(f'`--output_dir` is not provided. Results will be saved at {args.output_dir}')

    # load generation file
    with open(args.pred_file, 'r') as f:
        all_predictions = json.load(f)

    evaluator = Evaluator(nli_model_path=args.nli_model_path, device_map=args.device_map)

    all_scores = {}
    # evaluate prediction
    all_scores['predict'] = evaluator.evaluate_claimsplit(
        predictions=[sample['predict'] for sample in all_predictions],
        references=[sample['sentence'] for sample in all_predictions],
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        verbose=True
    )
    
    table = []
    for pred_name, scores in all_scores.items():
        table.append([pred_name] + list(scores.values()))
    
    headers = [''] + list(all_scores[pred_name].keys())
    headers = [header.replace(' ', '\n') for header in headers]
    table = tabulate.tabulate(table, headers=headers, tablefmt='grid')
    # print and save
    print(table)
    output_name = 'metrics_' + os.path.basename(args.pred_file).split('.')[0] + '.txt'
    with open(os.path.join(args.output_dir, output_name), 'w') as f:
        f.write(table)
    logger.info(f'Metrics saved at {os.path.join(args.output_dir, output_name)}')
        

if __name__ == '__main__':
    run_evaluation()