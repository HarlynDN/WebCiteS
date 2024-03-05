from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.integrations import is_deepspeed_zero3_enabled



def load_model_and_tokenizer(data_args, model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    load_args = {
        'low_cpu_mem_usage': False if is_deepspeed_zero3_enabled() else True, # low_cpu_mem_usage is not supported in zero3
    }

    if getattr(config, 'model_type', None) == 'chatglm':
        load_args['empty_init'] = False if is_deepspeed_zero3_enabled() else True
        
    if getattr(config, "is_encoder_decoder", False): # encoder-decoder model such as T5
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, **load_args)
    else: # decoder-only model such as GPT
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, **load_args)

    if training_args.do_train:
        model = model.float()
        
    tokenizer.label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if not getattr(config, "is_encoder_decoder", False):
        tokenizer.padding_side = 'left' # left padding for batch generation 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # add sep token
    if getattr(tokenizer, 'sep_token_id') is None:
        tokenizer.add_special_tokens({'sep_token': '<sep>'})

    if training_args.do_train:
        model.enable_input_require_grads()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    return model, tokenizer