from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


class AQFSTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ 
        Made some modifications to work with decoder-only models like GPT:
            - remove `labels` from inputs if predict_with_generate is True
            - remove input_ids from generated tokens if model is not encoder-decoder 
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        has_labels = "labels" in inputs
        # For encoder-decoder models, if the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if getattr(self.model.config, "is_encoder_decoder", False): 
            if (
                has_labels
                and "decoder_input_ids" in inputs
                and inputs["labels"].shape == inputs["decoder_input_ids"].shape
            ):
                labels = inputs['labels']
                inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
            else:
                labels = None
            loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

        # For decoder-only models like GPT
        else:
            # We need to remove original labels from inputs since it's different from input_ids
            labels = inputs.pop('labels') if has_labels else None
            loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
            # remove input_ids from generated_tokens
            generated_tokens = generated_tokens[:, inputs['input_ids'].shape[-1]:]

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        if has_labels:
            labels = labels.to(generated_tokens.device)
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        return loss, generated_tokens, labels


def prepare_trainer(
    model, 
    tokenizer,
    training_args, 
    train_dataset=None, 
    eval_dataset=None, 
    callbacks=None
):
    # data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.label_pad_token_id,
        pad_to_multiple_of=8,
        padding=True,
    )

    trainer = AQFSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    return trainer