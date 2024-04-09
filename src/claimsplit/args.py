import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser
)
from transformers.integrations import is_deepspeed_zero3_enabled

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "The directory of dataset, will be passed to `datasets.load_dataset`."
                  "If not provided, will look for `train_file`, `validation_file`, `test_file` instead."}
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    source_column: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the column in the datasets containing the input texts (for text generation task)."},
    )
    target_column: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the column in the datasets containing target texts (for text generation task)"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization. "
                  "Sequences longer than this will be truncated."},
    )
    max_target_length: Optional[int] = field(
        default=400,
        metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                  "than this will be truncated. This only takes effect for text generation."},
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached datasets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, 
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of eval examples to this value if set."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of predict examples to this value if set."},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, 
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to model path and configs (such as decoding strategy).
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

def parse_args():
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # args post init
    if 'baichuan2' in model_args.model_name_or_path.lower() and (training_args.do_eval or training_args.do_predict) and is_deepspeed_zero3_enabled():
        logger.warning(
            "Baichuan2 models currently do not support inference in DeepSpeed zero3, so evaluation/prediction are disabled. "
            "See https://github.com/baichuan-inc/Baichuan2/issues/39"
            )
        training_args.do_eval = False
        training_args.do_predict = False
        training_args.evaluation_strategy = "no"

    return data_args, model_args, training_args
       