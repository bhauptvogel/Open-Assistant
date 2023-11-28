from model_training.custom_datasets.rank_datasets import AnthropicRLHF, HellaSwagDataset, SHPDataset
from torch.utils.data import Dataset


def load_anthropic_rlhf(json_filename) -> tuple[Dataset, Dataset]:
    train = AnthropicRLHF(json_filename, split="train")
    validation = AnthropicRLHF(json_filename, split="test")
    return train, validation


def load_shp() -> tuple[Dataset, Dataset]:
    train = SHPDataset(split="train")
    validation = SHPDataset(split="validation")
    return train, validation


def load_hellaswag() -> tuple[Dataset, Dataset]:
    train = HellaSwagDataset(split="train")
    validation = HellaSwagDataset(split="validation")
    return train, validation
