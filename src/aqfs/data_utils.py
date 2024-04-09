import os
import json
from typing import Dict
import numpy as np  
import torch
import datasets

full_instruction ="给定一个问题和多条搜索结果, 你需要准确地理解问题的需求，将搜索结果整理总结成回答，并标注参考来源。请注意以下几点：\n\
1. 你的回答需要简洁清晰，逻辑连贯，内容全面，所有观点都需要被搜索结果中的内容所支持。\n\
2. 你的回答需要引用参考来源。你需要在每句话的结尾标注所参考的搜索结果的编号，放在[]中。你需要全面地引用所有能够支持这句话观点的搜索结果。\n\
3. 你需要将多个搜索结果中相似的观点或信息总结为一句话，并同时标注多个引用。"

short_instruction = "给定一个问题和多条搜索结果，请根据搜索结果整理总结出回答，并标注参考来源。\
回答需要简洁清晰、逻辑连贯并严格依照搜索结果。你需要将不同的观点分别总结成多句话，将相似的观点总结成一句话并同时标注多个来源。"

def load_dataset(data_args):
    if data_args.data_dir is not None:
        dataset =  datasets.load_dataset(data_args.data_dir)
    else:
        assert data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None, \
            "You need to provide at least one of `train_file`, `validation_file`, `test_file`"
        data_dict = {}
        if data_args.train_file is not None:
            data_dict['train'] = datasets.load_dataset(data_args.train_file.split('.')[-1], data_files=data_args.train_file)
        if data_args.validation_file is not None:
            data_dict['validation'] = datasets.load_dataset(data_args.validation_file.split('.')[-1], data_files=data_args.validation_file)
        if data_args.test_file is not None:
            data_dict['test'] = datasets.load_dataset(data_args.test_file.split('.')[-1], data_files=data_args.test_file)
        dataset = datasets.DatasetDict(data_dict)
    return dataset

def preprocess_causal(examples: Dict, tokenizer, data_args, fewshot_exemplar=None):
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
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            history = [(fewshot_exemplar[data_args.source_column], fewshot_exemplar[data_args.target_column])] if fewshot_exemplar else None
            prefix = data_args.task_prefix
            prompt = prefix+ query.strip() + '\n\n' if history is None else prefix + '\n\n'.join(history[0]) + '\n\n' + prefix + query.strip() + '\n\n'
            source_ids = tokenizer.encode(text=prompt, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=answer, truncation=True, max_length=data_args.max_target_length)
            context_length = len(source_ids)
            input_ids = source_ids + target_ids
            labels = [tokenizer.label_pad_token_id] * context_length + target_ids + [tokenizer.eos_token_id]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
    return model_inputs

def preprcocess_seq2seq(examples, tokenizer, data_args, fewshot_exemplar=None):
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
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            prefix = data_args.task_prefix
            history = [(fewshot_exemplar[data_args.source_column], fewshot_exemplar[data_args.target_column])] if fewshot_exemplar else None
            prompt = prefix + query.strip() + '\n\n' if history is None else prefix + '\n\n'.join(history[0]) + '\n\n' + prefix + query.strip() + '\n\n'
            source_ids = tokenizer.encode(text=prompt, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=answer, truncation=True, max_length=data_args.max_target_length)
            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)
    return model_inputs

def preprocess_train_chatglm2(examples, tokenizer, data_args, fewshot_exemplar=None):
    """
    `preprocess_causal` using `tokenizer.build_prompt` from ChatGLMTokenizer
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            history = [(fewshot_exemplar[data_args.source_column], fewshot_exemplar[data_args.target_column])] if fewshot_exemplar else None
            prompt = tokenizer.build_prompt(query, history=history)
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, max_length=data_args.max_target_length-1)
            context_length = len(source_ids)
            input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.label_pad_token_id] * context_length + target_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
    return model_inputs

def preprocess_eval_chatglm2(examples, tokenizer, data_args, fewshot_exemplar=None):
    """
    `preprocess_seq2seq` using `tokenizer.build_prompt` from ChatGLMTokenizer
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            history = [(fewshot_exemplar[data_args.source_column], fewshot_exemplar[data_args.target_column])] if fewshot_exemplar else None
            prompt = tokenizer.build_prompt(query, history=history)
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, max_length=data_args.max_target_length)
            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)
    return model_inputs

def preprocess_train_chatglm3(examples, tokenizer, data_args, fewshot_exemplar=None):
    """
    `preprocess_causal` using `tokenizer.build_chat_input` from ChatGLMTokenizer
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            prompt, target = examples[data_args.source_column][i], examples[data_args.target_column][i]
            if fewshot_exemplar:
                prompt = fewshot_exemplar[data_args.source_column] + '\n\n' + fewshot_exemplar[data_args.target_column] + '\n\n' + prompt
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=target, add_special_tokens=False, truncation=True, max_length=data_args.max_target_length-1)
            context_length = len(source_ids)
            input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.label_pad_token_id] * context_length + target_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
    return model_inputs

def preprocess_eval_chatglm3(examples, tokenizer, data_args, fewshot_exemplar=None):
    """
    `preprocess_seq2seq` using `tokenizer.build_prompt` from ChatGLMTokenizer
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            prompt, target = examples[data_args.source_column][i], examples[data_args.target_column][i]
            if fewshot_exemplar:
                prompt = fewshot_exemplar[data_args.source_column] + '\n\n' + fewshot_exemplar[data_args.target_column] + '\n\n' + prompt
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=data_args.max_source_length)
            target_ids = tokenizer.encode(text=target, add_special_tokens=False, truncation=True, max_length=data_args.max_target_length)
            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)
    return model_inputs

def preprocess_chat_chatglm3(examples, tokenizer, data_args, fewshot_exemplar=None):
    """
    `preprocess_seq2seq` based on `tokenizer.build_chat_input` from ChatGLMTokenizer
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        prompt, target = examples[data_args.source_column][i], examples[data_args.target_column][i]
        history = []
        if fewshot_exemplar:
            history.append({"role": "user", "content": fewshot_exemplar[data_args.source_column]})
            history.append({"role": "assistant", "content": fewshot_exemplar[data_args.target_column]})
        # based on ChatGLMTokenizer.build_chat_input
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(tokenizer.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(tokenizer.build_single_message('user', "", prompt))
        input_ids = input_ids[:data_args.max_source_length-1]
        input_ids.extend([tokenizer.get_command("<|assistant|>")])

        target_ids = tokenizer.encode(text=target, add_special_tokens=False, truncation=True, max_length=data_args.max_target_length-1)
        attention_mask = [1] * len(input_ids)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(target_ids)
    return model_inputs

def _preprocess_baichuan2(conversations, tokenizer, max_source_length, max_target_length, user_tokens=[195], assistant_tokens=[196], is_eval=False):
    """
    based on https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py
    """
    if is_eval:
        assert conversations[-1]['from'] == 'assistant'
        target_text = conversations.pop()['value']
    input_ids = []
    labels = []
    for message in conversations:
        from_ = message["from"]
        value = message["value"]
        value_ids = tokenizer.encode(value)
        if from_ == "human":
            input_ids +=  user_tokens + value_ids
            labels += [tokenizer.eos_token_id] + [tokenizer.label_pad_token_id] * len(
                value_ids
            )
        else:
            input_ids += assistant_tokens + value_ids
            labels += [tokenizer.label_pad_token_id] + value_ids

    max_seq_length = max_source_length + max_target_length

    if is_eval: # similar to `preprocess_seq2seq`
        input_ids = input_ids[: max_source_length-1]
        input_ids.extend(assistant_tokens)
        labels = tokenizer.encode(target_text, add_special_tokens=False, truncation=True, max_length=max_target_length-1)
        labels.append(tokenizer.eos_token_id)
    else: # similar to `preprocess_causal`
        input_ids = input_ids[: max_seq_length-1]
        input_ids.append(tokenizer.eos_token_id)
        labels = labels[: max_seq_length-1]    
        labels.append(tokenizer.eos_token_id)

    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, attention_mask, labels

def preprocess_train_baichuan2(examples, tokenizer, data_args, fewshot_exemplar=None):
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            conversations = []
            if fewshot_exemplar:
                conversations.append({"from": "human", "value": fewshot_exemplar[data_args.source_column]})
                conversations.append({"from": "assistant", "value": fewshot_exemplar[data_args.target_column]})
            conversations.append({"from": "human", "value": query})
            conversations.append({"from": "assistant", "value": answer})
            input_ids, attention_mask, labels = _preprocess_baichuan2(conversations, tokenizer, data_args.max_source_length, data_args.max_target_length)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
    return model_inputs

def preprocess_eval_baichuan2(examples, tokenizer, data_args, fewshot_exemplar=None):
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[data_args.source_column])):
        if examples[data_args.source_column][i] and examples[data_args.target_column][i]:
            query, answer = examples[data_args.source_column][i], examples[data_args.target_column][i]
            conversations = []
            if fewshot_exemplar:
                conversations.append({"from": "human", "value": fewshot_exemplar[data_args.source_column]})
                conversations.append({"from": "assistant", "value": fewshot_exemplar[data_args.target_column]})
            conversations.append({"from": "human", "value": query})
            conversations.append({"from": "assistant", "value": answer})
            input_ids, attention_mask, labels = _preprocess_baichuan2(conversations, tokenizer, data_args.max_source_length, data_args.max_target_length, is_eval=True)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
    return model_inputs


PREPROCESS_FUNCTION_DICT = {
    'causal': {
        'train': preprocess_causal,
        'eval': preprcocess_seq2seq
    },
    'seq2seq':{
        'train': preprcocess_seq2seq,
        'eval': preprcocess_seq2seq
    },
    'chatglm2': {
        'train': preprocess_train_chatglm2,
        'eval': preprocess_eval_chatglm2,
        'chat': preprocess_eval_chatglm2
    },
    'chatglm3': {
        'train': preprocess_train_chatglm3,
        'eval': preprocess_eval_chatglm3,
        'chat': preprocess_chat_chatglm3
    },
    'baichuan': {
        'train': preprocess_train_baichuan2,
        'eval': preprocess_eval_baichuan2,
        'chat': preprocess_eval_baichuan2
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
    model_type = getattr(config, "model_type", None)
    if model_type == 'chatglm':
        if 'chatglm2-6b' in getattr(config, "_name_or_path", None):
            model_type = 'chatglm2'
        else:
            model_type = 'chatglm3'

    if model_type in ['t5', 'mt5']:
        data_args.task_prefix = 'summarize: '
    else:
        data_args.task_prefix = ''

    if model_type in PREPROCESS_FUNCTION_DICT.keys():
        preproc_train_func = PREPROCESS_FUNCTION_DICT[model_type]['train']
        if getattr(training_args, "predict_with_generate", False):
            if data_args.use_chat_format:
                preproc_eval_func = PREPROCESS_FUNCTION_DICT[model_type]['chat']
            else:
                preproc_eval_func = PREPROCESS_FUNCTION_DICT[model_type]['eval']  
        else:
            preproc_eval_func = preproc_train_func
    else:
        if getattr(config, "is_encoder_decoder", False): # encoder-decoder model such as T5
            preproc_train_func = PREPROCESS_FUNCTION_DICT['seq2seq']['train']
            preproc_eval_func = PREPROCESS_FUNCTION_DICT['seq2seq']['eval']
        else: # decoder-only model such as GPT
            preproc_train_func = PREPROCESS_FUNCTION_DICT['causal']['train']
            preproc_eval_func = PREPROCESS_FUNCTION_DICT['causal']['eval'] if getattr(training_args, "predict_with_generate", False) else preproc_train_func

    fewshot_exemplar = None
    if data_args.exemplar_id is not None:
        for sample in dataset['train']:
            if sample['id'] == data_args.exemplar_id:
                fewshot_exemplar = sample
                fewshot_exemplar[data_args.source_column] = full_instruction + fewshot_exemplar[data_args.source_column].lstrip(short_instruction)
                break

    def preprocess_train(examples):
        return preproc_train_func(examples, tokenizer, data_args, fewshot_exemplar)
    
    def preprocess_eval(examples):
        return preproc_eval_func(examples, tokenizer, data_args, fewshot_exemplar)

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
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        label_ids = np.where(predict_results.label_ids != tokenizer.label_pad_token_id, predict_results.label_ids, tokenizer.pad_token_id)
        predictions = np.array([pred.strip() for pred in predictions]).reshape(label_ids.shape[0], -1)
        labels = tokenizer.batch_decode(
            label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        labels = [label.strip() for label in labels]
        ouputs = []
        for pred, lbl, sample in zip(predictions, labels, test_dataset):
            item = {}
            if data_args.query_column in sample:
                item['query'] = sample[data_args.query_column]
            if data_args.doc_column in sample:
                item['docs'] = sample[data_args.doc_column]
            item['label'] = lbl
            item['predict'] = list(pred)
            ouputs.append(item)
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            json.dump(ouputs, writer, indent=2, ensure_ascii=False)
        print(f"Predict results saved to {output_prediction_file}")