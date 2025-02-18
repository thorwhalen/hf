"""Base functionality for hf"""

import os
from collections.abc import Mapping
from datasets import load_dataset
from transformers import AutoModel
from huggingface_hub import (
    scan_cache_dir,
    snapshot_download,
    list_models,
    model_info,
    list_datasets,
    dataset_info,
)
from huggingface_hub.hf_api import DatasetInfo, ModelInfo


# TODO: Design flexible way to get keys, or info, or data files (mapping) or loaded
#  item (dataset/model), with mapping interface for all. Use model_info and dataset_info for info


def ensure_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath


def list_local_datasets():
    """
    Dynamically list locally cached datasets using scan_cache_dir.
    Returns a list of repo IDs for repositories of type "dataset".
    """
    cache_info = scan_cache_dir()
    return [repo.repo_id for repo in cache_info.repos if repo.repo_type == "dataset"]


def list_local_models():
    """
    Dynamically list locally cached models using scan_cache_dir.
    Returns a list of repo IDs for repositories of type "model".
    """
    cache_info = scan_cache_dir()
    return [repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"]


class HfDatasets(Mapping):
    def __getitem__(self, key):
        if isinstance(key, DatasetInfo):
            key = key.id
        return load_dataset(key)
    
    def _keys(self):
        return list_local_datasets()

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    @staticmethod
    def search(filter, **kwargs):
        """
        Return a list of datasets that match the query substring.
        """
        return list_datasets(filter=filter, **kwargs)


class HfModels(Mapping):
    def _keys(self):
        return list_local_models()

    def __getitem__(self, key):
        if isinstance(key, ModelInfo):
            key = key.id
        return snapshot_download(repo_id=key)

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(list(self))

    def search(self, filter, **kwargs):
        """
        Return a list of model repo IDs (from the local cache) that match the query substring.
        """
        return list_models(filter=filter, **kwargs)
