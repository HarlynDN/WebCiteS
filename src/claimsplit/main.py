import os
import json
from transformers import set_seed
from dataclasses import asdict

from args import parse_args
from data_utils import load_dataset, preprocess_dataset, save_predictions
from model_utils import load_model_and_tokenizer
from trainer_utils import prepare_trainer

def main():
    data_args, model_args, training_args = parse_args()
    set_seed(training_args.seed)
    # load dataset
    dataset = load_dataset(data_args)
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        data_args=data_args, 
        model_args=model_args, 
        training_args=training_args
    )
    
    # data preprocessing
    train_dataset, eval_dataset, predict_dataset = preprocess_dataset(
        dataset=dataset, 
        config=model.config, 
        tokenizer=tokenizer, 
        data_args=data_args,
        training_args=training_args
    )

    # Load trainer
    trainer = prepare_trainer(
        model=model, 
        tokenizer=tokenizer, 
        training_args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # save hparams
        if trainer.is_world_process_zero():
            hparams = {
                'data_args': asdict(data_args),
                'model_args': asdict(model_args),
                'training_args': asdict(training_args),
            }
            with open(os.path.join(training_args.output_dir, "train_hparams.json"), 'w') as f:
                json.dump(hparams, f, indent=2)

    # Inference
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics

        # save results
        if trainer.is_world_process_zero():
            save_predictions(
                predict_results=predict_results,
                test_dataset=dataset['test'],
                tokenizer=tokenizer, 
                data_args=data_args,
                training_args=training_args,
            )
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
        
if __name__ == '__main__':
    main()
