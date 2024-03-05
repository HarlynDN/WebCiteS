import os
import json
from typing import Dict, List, Optional, Tuple, Union  
import numpy as np  
import datasets

def load_dataset(data_args):
    dataset =  datasets.load_dataset(data_args.data_dir)
    return dataset

def preprocess_causal(examples: Dict, tokenizer, data_args):
    """
    Tokenize the examples, concatenate the source ids and target ids as labels
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            prompt, target = examples[data_args.source_column][i], examples[data_args.target_column][i]
            prompt = data_args.task_prefix + prompt
            if isinstance(target, list):
                target = tokenizer.sep_token.join(target)
            source_ids = tokenizer.encode(text=prompt, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=target, truncation=True, max_length=data_args.max_target_length) 
            context_length = len(source_ids)
            input_ids = source_ids + target_ids
            labels = [tokenizer.label_pad_token_id] * context_length + target_ids + [tokenizer.eos_token_id]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
    return model_inputs


def preprcocess_seq2seq(examples, tokenizer, data_args):
    """
    Tokenize the examples, take the source ids as input_ids, and target ids as labels
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            prompt, target = examples[data_args.source_column][i], examples[data_args.target_column][i]
            prompt = data_args.task_prefix + prompt
            if isinstance(target, list):
                target = tokenizer.sep_token.join(target)
            source_ids = tokenizer.encode(text=prompt, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=target, truncation=True, max_length=data_args.max_target_length)
            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)
    return model_inputs


PREPROCESS_FUNCTION_DICT = {
    'causal': {
        'train': preprocess_causal,
        'eval': preprcocess_seq2seq
    },
    'seq2seq':{
        'train': preprcocess_seq2seq,
        'eval': preprcocess_seq2seq
    }
}

def preprocess_dataset(dataset, config, tokenizer, data_args, training_args):
    def print_dataset_example(example):
        if training_args.local_rank < 1: # print only in main process
            for k, v in example.items():
                if "input_ids" in k :
                    print(f"{k}==>\n{v}")
                    print(tokenizer.decode(v))
                if "labels" in k: 
                    print(f"{k}==>\n{v}")
                    labels = np.where(np.array(v) != -100, np.array(v), tokenizer.pad_token_id)
                    print(tokenizer.decode(labels, skip_special_tokens=True))
    
    if getattr(config, "is_encoder_decoder", False): # encoder-decoder model such as T5
        preproc_train_func = PREPROCESS_FUNCTION_DICT['seq2seq']['train']
        preproc_eval_func = PREPROCESS_FUNCTION_DICT['seq2seq']['eval']
    else: # decoder-only model such as GPT
        preproc_train_func = PREPROCESS_FUNCTION_DICT['causal']['train']
        preproc_eval_func = PREPROCESS_FUNCTION_DICT['causal']['eval'] if getattr(training_args, "predict_with_generate", False) else preproc_train_func

    data_args.task_prefix = 'claim-split: '


    def preprocess_train(examples):
        return preproc_train_func(examples, tokenizer, data_args)
    
    def preprocess_eval(examples):
        return preproc_eval_func(examples, tokenizer, data_args)

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=dataset['train'].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])
    
    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=dataset['validation'].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )
        print_dataset_example(eval_dataset[0])
    
    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="predict dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=dataset['test'].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on predict dataset",
            )
        print_dataset_example(predict_dataset[0])

    return train_dataset, eval_dataset, predict_dataset
    
def save_predictions(predict_results, test_dataset, tokenizer, data_args, training_args):
    # save model predictions
    if training_args.predict_with_generate:
        predictions = predict_results.predictions.reshape(-1, predict_results.predictions.shape[-1])
        predictions = np.where(predictions != tokenizer.label_pad_token_id, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        label_ids = np.where(predict_results.label_ids != tokenizer.label_pad_token_id, predict_results.label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(
            label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        predictions = [[pred.strip() for pred in preds.strip(tokenizer.pad_token).strip(tokenizer.eos_token).split(tokenizer.sep_token)] for preds in predictions] 
        labels = [[lbl.strip() for lbl in lbls.strip(tokenizer.pad_token).strip(tokenizer.eos_token).split(tokenizer.sep_token)] for lbls in labels]

        ouputs = []
        for pred, lbl, sample in zip(predictions, labels, test_dataset):
            item = {}
            if 'summary' in sample:
                item['summary'] = sample['summary']
            item[data_args.source_column] = sample[data_args.source_column]
            item['label'] = lbl
            item['predict'] = list(pred)
            ouputs.append(item)
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            json.dump(ouputs, writer, indent=2, ensure_ascii=False)
        print(f"Predict results saved to {output_prediction_file}")