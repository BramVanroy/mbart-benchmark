from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def make_datasets(input_seq_length: int, output_seq_length: int, train_samples: int, eval_samples: int, tokenizer: PreTrainedTokenizer):
    voc_size = len(tokenizer)
    en_token_id = tokenizer.convert_tokens_to_ids("en_XX")
    meta = torch.LongTensor([tokenizer.eos_token_id, en_token_id])

    train_data = defaultdict(list) if train_samples else None
    if train_samples:
        meta_train = meta.repeat(train_samples, 1)
        # low=10 to not have special tokens like SOS/EOS
        input_ids = torch.randint(low=10, high=voc_size-1, size=(train_samples, input_seq_length-2))
        input_ids = torch.cat((input_ids, meta_train), dim=1).tolist()

        # Iterate batch to process each sequence separately
        for ids in input_ids:
            for key, values in tokenizer.prepare_for_model(ids, return_tensors="pt").items():
                train_data[key].append(values)

        train_data = dict(train_data)

        labels = torch.randint(low=10, high=voc_size-1, size=(train_samples, output_seq_length-2))
        lang_codes = torch.full((train_samples, 1), en_token_id)
        eos = torch.full((train_samples, 1), tokenizer.eos_token_id)
        labels = torch.cat((lang_codes, labels, eos), dim=1)
        train_data["labels"] = labels

        train_data = DummyDataset(train_data)

    eval_data = defaultdict(list) if eval_samples else None
    if eval_samples:
        meta_eval = meta.repeat(eval_samples, 1)
        # low=10 to not have special tokens like SOS/EOS
        input_ids = torch.randint(low=10, high=voc_size-1, size=(eval_samples, input_seq_length-2))
        input_ids = torch.cat((input_ids, meta_eval), dim=1).tolist()

        # Iterate batch to process each sequence separately
        for ids in input_ids:
            for key, values in tokenizer.prepare_for_model(ids, return_tensors="pt").items():
                eval_data[key].append(values)

        eval_data = dict(eval_data)

        labels = torch.randint(low=10, high=voc_size-1, size=(eval_samples, output_seq_length-2))
        lang_codes = torch.full((eval_samples, 1), en_token_id)
        eos = torch.full((eval_samples, 1), tokenizer.eos_token_id)
        labels = torch.cat((lang_codes, labels, eos), dim=1)
        eval_data["labels"] = labels

        eval_data = DummyDataset(eval_data)

    return train_data, eval_data


class DummyDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.data_size = len(data_dict[list(data_dict.keys())[0]])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data = {}
        for key, values in self.data.items():
            data[key] = values[idx].clone()

        return data
