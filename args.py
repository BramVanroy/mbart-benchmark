from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/mbart-large-cc25",
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        },
    )
    input_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model. Batches will be padded up to max."
                " length in the batch."
            )
        },
    )
    output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total output sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model. Batches will be padded up to max."
                " length in the batch."
            )
        },
    )
    train_samples: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set per given language."
            )
        },
    )
    eval_samples: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set per given language."
            )
        },
    )
    num_additional_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "In your own model you may have added some additional tokens, which increases the embedding size. Here"
                " you can specify how many to add. Random tokens will be added to mimic correct memory usage."
            )
        },
    )