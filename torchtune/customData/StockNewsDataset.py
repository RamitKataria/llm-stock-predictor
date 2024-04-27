from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from torchtune.data import InstructTemplate
from typing import Any, Dict, Mapping, Optional
from torchtune.modules.tokenizers import Tokenizer

class StockNewsDataset(Dataset):
    """
    Class to support stock news dataset for predicting stock price changes based on news articles,
    formatted for a regression task.
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        prompt = self.template.format(sample, self._column_map)
        input_ids = self._tokenizer.encode(prompt)

        # Extract and convert stock price changes to a tensor for regression
        labels = self._prepare_labels(sample, self._column_map)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _prepare_labels(self, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]]) -> torch.Tensor:
        # This method extracts stock price changes as labels for regression
        stock_changes = [sample.get(column_map.get(f"Company{i}", f"Company{i}"), 0.0) for i in range(1, 501)]
        return torch.tensor(stock_changes, dtype=torch.float)
