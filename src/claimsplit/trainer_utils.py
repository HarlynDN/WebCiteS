from transformers import (
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq, 
)

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    return trainer