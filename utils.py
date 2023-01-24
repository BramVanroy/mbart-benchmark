import logging
import string
import random
import sys

import transformers

from transformers import AutoTokenizer, MBartForConditionalGeneration

logger = logging.getLogger(__name__)


def set_logging(logger, log_level):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_tokenizer_and_model(model_name: str, num_additional_tokens: int = 0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    if num_additional_tokens:
        voc = set(tokenizer.get_vocab().keys())

        to_add = set([])
        while len(to_add) < num_additional_tokens:
            random_token = ''.join(random.choices(string.ascii_lowercase, k=5))

            if random_token not in voc and random_token not in to_add:
                to_add.add(random_token)

        tokenizer.add_tokens(list(sorted(to_add)))
        logger.info(f"Added {len(to_add):,} new tokens!")

        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
