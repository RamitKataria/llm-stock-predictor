
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, InstructTemplate, Message

from torchtune.modules.tokenizers import Tokenizer


class StockNewsDataset(Dataset):
    """
    Class to support stock news dataset for stock price prediction based on news articles.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: InstructTemplate,
        column_map: Optional[Dict[str, str]] = None,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.template = template
        self._column_map = column_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = self.template.format(sample, self._column_map)
        input_ids = self._tokenizer.encode(prompt)

        # Assume labels are being predicted directly from the news impact (this will need customization)
        labels = self._prepare_labels(sample, self._column_map)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _prepare_labels(self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]]):
        # This method needs to extract and encode stock price changes as labels
        stock_changes = [sample.get(column_map.get(f"Company{i}", f"Company{i}"), 0) for i in range(1, 501)]
        return self._tokenizer.encode(stock_changes)
