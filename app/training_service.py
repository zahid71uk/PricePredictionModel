import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI

from app.model_service import ModelService
from app.training import train_model

logger = logging.getLogger(__name__)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, path = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(descriptor)
    return Path(path)


def promote_dataset(upload: bytes, application: FastAPI) -> tuple[dict, int]:
    with application.state.training_lock:
        return _promote_dataset_locked(upload, application)


def _promote_dataset_locked(upload: bytes, application: FastAPI) -> tuple[dict, int]:
    dataset_path = Path(application.state.dataset_path)
    model_path = Path(application.state.model_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_dataset = _temporary_path(dataset_path.parent, ".csv")
    candidate_model = _temporary_path(model_path.parent, ".joblib")
    backup_dataset = _temporary_path(dataset_path.parent, ".csv.bak")
    backup_model = _temporary_path(model_path.parent, ".joblib.bak")
    dataset_promoted = False
    model_promoted = False

    try:
        candidate_dataset.write_bytes(upload)
        metrics = train_model(candidate_dataset, candidate_model)
        candidate_service = ModelService.load(candidate_model)
        dataset_count = metrics["train_samples"] + metrics["test_samples"]

        with application.state.dataset_lock:
            had_dataset = dataset_path.is_file()
            had_model = model_path.is_file()
            if had_dataset:
                shutil.copy2(dataset_path, backup_dataset)
            if had_model:
                shutil.copy2(model_path, backup_model)
            try:
                os.replace(candidate_dataset, dataset_path)
                dataset_promoted = True
                os.replace(candidate_model, model_path)
                model_promoted = True
            except OSError:
                if dataset_promoted:
                    if had_dataset:
                        os.replace(backup_dataset, dataset_path)
                    else:
                        dataset_path.unlink(missing_ok=True)
                if model_promoted:
                    if had_model:
                        os.replace(backup_model, model_path)
                    else:
                        model_path.unlink(missing_ok=True)
                raise
            application.state.model_service = candidate_service

        return metrics, dataset_count
    finally:
        for path in (candidate_dataset, candidate_model, backup_dataset, backup_model):
            path.unlink(missing_ok=True)