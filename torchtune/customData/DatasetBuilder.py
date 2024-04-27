# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from torchtune.data import StockNewsTemplate
from torchtune.datasets._preference import StockNewsDataset
from torchtune.modules.tokenizers import Tokenizer


def stock_news_regression_dataset(
    tokenizer: Tokenizer,
    source: str = "./data/dataset.csv",  # Adjusted to the name of your CSV file
    max_seq_len: int = 512,  # Adjust based on your model's capacity
    data_dir: str = "data/stock_news",  # Directory to store processed datasets
    train_test_split: float = 0.8  # Proportion of data to use for training
) -> dict:
    """
    Dataset builder for stock news dataset, formatted for regression tasks involving news impact on stock prices.
    This function splits the dataset into train and test sets programmatically.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data.
        source (str): Path string or identifier of the dataset, supported by Hugging Face's `load_dataset`.
        max_seq_len (int): Maximum number of tokens in the returned input token id lists.
        data_dir (str): Local directory path to store and manage dataset files.
        train_test_split (float): Proportion of data to be used for training (the rest for testing).

    Returns:
        dict: A dictionary containing 'train' and 'test' datasets.
    """
    # Dynamically generate column mappings for the company stock changes
    column_map = {
        "news": "News",
        "date": "Date",
        "sentiment": "Sentiment",
        "industry": "Industry",
        **{f"Company{i}": f"Company{i}" for i in range(1, 501)}
    }

    # Load the dataset from a CSV file using the Hugging Face 'datasets' library
    dataset = load_dataset('csv', data_files=source, cache_dir=data_dir)

    # Split the dataset into train and test sets
    train_dataset = dataset['train'].train_test_split(train_size=train_test_split)['train']
    test_dataset = dataset['train'].train_test_split(train_size=train_test_split)['test']

    return {
        'train': StockNewsDataset(
            tokenizer=tokenizer,
            source=train_dataset,
            template=StockNewsTemplate(),
            column_map=column_map,
            max_seq_len=max_seq_len
        ),
        'test': StockNewsDataset(
            tokenizer=tokenizer,
            source=test_dataset,
            template=StockNewsTemplate(),
            column_map=column_map,
            max_seq_len=max_seq_len
        )
    }
