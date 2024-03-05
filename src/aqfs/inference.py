import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    set_seed,
    logging
)
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from args import parse_args
from data_utils import load_dataset, preprocess_dataset


def main():
    data_args, model_args, training_args = parse_args()
    set_seed(training_args.seed)

    # load dataset
    dataset = load_dataset(data_args)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    load_args = {"device_map": "auto", 'torch_dtype': config.torch_dtype} # shard the model weights across devices

    if getattr(config, 'model_type', None) == 'chatglm':
        load_args['empty_init'] = True
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, **load_args)
    
    # tokenizer post init
    tokenizer.label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if not getattr(config, "is_encoder_decoder", False):
        tokenizer.padding_side = 'left' # left padding for batch generation 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # data preprocessing
    train_dataset, eval_dataset, predict_dataset = preprocess_dataset(
        dataset=dataset, 
        config=config, 
        tokenizer=tokenizer, 
        data_args=data_args,
        training_args=training_args
    )
    # dataloader
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.label_pad_token_id,
        pad_to_multiple_of=8,
        padding=True,
    )
    predict_dataloader = DataLoader(
        predict_dataset, 
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    # batch inference
    generation_kwargs = {
        "do_sample": model_args.do_sample, 
        "top_p": model_args.top_p, 
        "max_new_tokens": data_args.max_target_length,
        "temperature": model_args.temperature
    }
    
    model.eval()
    predictions = []
    logger.info("***** Running Generation *****")
    for idx, batch in tqdm(enumerate(predict_dataloader), total=len(predict_dataloader)):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch['input_ids'].to(model.device), 
                attention_mask=batch['attention_mask'].to(model.device),
                **generation_kwargs
            )
        # remove input_ids from generated_tokens
        outputs = outputs[:, batch['input_ids'].shape[-1]:]
        predictions.extend(outputs.tolist())

    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # save model predictions
    predictions = np.array([pred.strip() for pred in predictions])
    ouputs = []
    for idx, pred in enumerate(predictions):
        sample = dataset['test'][idx]
        item = {}
        if data_args.query_column in sample:
            item['query'] = sample[data_args.query_column]
        if data_args.doc_column in sample:
            item['docs'] = sample[data_args.doc_column]
        item['label'] = sample[data_args.target_column]
        item['predict'] = [pred] if isinstance(pred, str) else list(pred)
        ouputs.append(item)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        json.dump(ouputs, writer, indent=2, ensure_ascii=False)
    print(f"Predict results saved to {output_prediction_file}")

if __name__ == '__main__':
    main()