import logging
import os
import sys

import evaluate
import numpy as np
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

from args import ModelArguments
from convert_results_readable import make_human_readable
from dataset import make_datasets
from utils import set_logging, get_tokenizer_and_model

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))

    if sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

        # Now, allow additional overrides
        if len(sys.argv) > 2:
            other_args = iter(sys.argv[2:])

            try:
                for arg in other_args:
                    if not arg.startswith("--"):
                        raise ValueError
                    arg = arg[2:]  # remove --
                    value = next(other_args)

                    for parsed_args in (model_args, training_args):
                        if hasattr(parsed_args, arg):
                            setattr(parsed_args, arg, value)
            except:
                raise ValueError("If additional arguments are given to the JSON file, they have to be in the format '--argument value'")
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    set_logging(logger, training_args.get_process_log_level())

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer, model = get_tokenizer_and_model(model_args.model_name_or_path, model_args.num_additional_tokens)

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    train_dataset, eval_dataset = make_datasets(model_args.input_seq_length,
                                                model_args.output_seq_length,
                                                model_args.train_samples,
                                                model_args.eval_samples,
                                                tokenizer)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        if trainer.is_world_process_zero():
            make_human_readable(f"{training_args.output_dir}/train_results.json",
                                to_add={"batch_size": trainer._train_batch_size})  # Save batch_size after auto_find

    max_length = training_args.generation_max_length
    num_beams = training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero():
            make_human_readable(f"{training_args.output_dir}/eval_results.json")


if __name__ == '__main__':
    main()
