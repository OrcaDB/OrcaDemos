import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy


def load_jsonl_data(file_path: Path, max_items: Optional[int] = None) -> List[Dict]:
    """Load and shuffle data from a JSONL file.

    :param file_path: Path to the JSONL file.
    :param max_items: Maximum number of items to load. Load all if None.
    :return: List of data entries.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    random.seed(10)
    random.shuffle(data)
    return data if max_items is None else data[:max_items]


def split_data(
    data: List[Any], train_ratio: float = 0.8, max_val_items: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """Split the data into training and validation sets.

    :param data: The complete dataset.
    :param train_ratio: The ratio of data to be used for training.
    :param max_val_items: The maximum number of validation items. Use all if None.
    :return: A tuple containing the training and validation sets.
    """
    total_size = len(data)

    if total_size < 2:
        raise ValueError("The dataset must contain at least 2 items to be split into training and validation sets.")

    val_size = int(total_size * (1 - train_ratio))

    if max_val_items is not None:
        val_size = min(val_size, max_val_items)

    # Ensure at least one item in the validation set
    val_size = max(val_size, 1)

    # Calculate the training size ensuring there's at least one training item
    train_size = total_size - val_size

    if train_size < 1:
        raise ValueError("The dataset split results in no training data. Adjust the train_ratio or max_val_items.")

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]

    return train_data, val_data


def load_finetuning_data(file_path: Path, max_items: Optional[int] = None) -> List[dspy.Example]:
    """Convert the JSONL data into training dataset of dspy.Example type for finetuning

    Args:
        file_path (Path): Jsonl dataset file path
        max_items (Optional[int], optional): total number of training dataset to be loaded. Defaults to None.

    Returns:
        List[dspy.Example]: Returns devset datasets
    """
    dataset = load_jsonl_data(file_path, max_items)
    return [dspy.Example(r).with_inputs("question") for r in dataset]
