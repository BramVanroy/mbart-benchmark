import dataclasses
import logging
import os
import sys
from pathlib import Path

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

    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    if config_file:
        config_args = parser.parse_json_file(json_file=os.path.abspath(config_file))

        # Find all other CLI arguments but exclude the current config file and the current Python file
        # We need Path(arg).name because on some systems this is automatically expanded
        # e.g., from `myscript.py` becomes `.\\myscript.py` in sys.argv. So we explicitly check for the name
        other_args = [arg for arg in sys.argv if arg != config_file and Path(arg).name != Path(__file__).name]

        # Find the argument names on CLI, i.e., those starting with -- (e.g., `output_dir`, or `fp16`)
        # arg[2:] to remove "--"
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}

        # Get the required arguments from the parser so that we can generate dummy values
        # This is needed for the next step, otherwise we get "xxx is a required argument" if we
        # do not specify the required argument on the CLI
        # We assume here that all "required" fields require one single argument, here just a dummy
        # Do NOT generate the dummy argument if it is already given as an arg in arg_names
        required_args = [(act.option_strings[0], "dummy")
                         for act in parser._actions
                         if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten

        # Parse the `cli_args` (actual CLI args + dummy required args) into dataclasses
        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        # Iterate over couples of dataclasses, where the first one is the initial one from config
        # and the second one is the one with CLI args + dummy required arguments
        # In this loop, we replace values in cfg_dc with the ones from the CLI
        # but only if they were _real_ CLI arguments (not the required dummies we generated)
        all_args = []
        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Filter the loaded `other_args` to only the arguments that were specified on the command-line (`arg_names`)
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names}
            # Replace the values in the config-loaded args with the ones loaded from cli
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))

        model_args, training_args = all_args
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
