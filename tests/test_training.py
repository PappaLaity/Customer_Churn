"""Unit tests for src.training.train.register_best_model.

These tests exercise the model-registry promotion policy without touching a live
MLflow server: MlflowClient is replaced with a MagicMock and we assert on the
stage transition the function requests.
"""
from unittest.mock import MagicMock

import pytest

from src.training import train


def _best_run():
    return {
        "model_name": "Random Forest",
        "test_recall": 0.83,
        "cv_mean": 0.79,
        "run_id": "abc123",
    }


def _fake_version(version="1"):
    v = MagicMock()
    v.version = version
    return v


@pytest.fixture
def mock_client(mocker):
    """Patch MlflowClient in the train module and return the instance mock."""
    client = MagicMock()
    client.create_model_version.return_value = _fake_version("1")
    mocker.patch.object(train, "MlflowClient", return_value=client)
    return client


def _prod_version(recall):
    v = MagicMock()
    v.current_stage = "Production"
    v.tags = {"test_recall": str(recall)}
    return v


def test_promotes_to_production_when_none_exists(mock_client):
    # No existing versions -> first model goes straight to Production.
    mock_client.search_model_versions.return_value = []

    version = train.register_best_model(_best_run(), model_registry_name="M")

    assert version == "1"
    mock_client.create_model_version.assert_called_once()
    call = mock_client.transition_model_version_stage.call_args
    assert call.kwargs["stage"] == "Production"
    assert call.kwargs["archive_existing_versions"] is True


def test_better_candidate_passes_gate_to_production(mock_client):
    # Incumbent recall 0.70 < candidate 0.83 -> gate passes, promote to Production.
    mock_client.search_model_versions.return_value = [_prod_version(0.70)]

    train.register_best_model(_best_run(), model_registry_name="M")

    call = mock_client.transition_model_version_stage.call_args
    assert call.kwargs["stage"] == "Production"
    assert call.kwargs["archive_existing_versions"] is True


def test_worse_candidate_fails_gate_and_stays_in_staging(mock_client):
    # Incumbent recall 0.95 > candidate 0.83 -> gate fails, land in Staging and
    # leave the existing Production version untouched.
    mock_client.search_model_versions.return_value = [_prod_version(0.95)]

    train.register_best_model(_best_run(), model_registry_name="M")

    call = mock_client.transition_model_version_stage.call_args
    assert call.kwargs["stage"] == "Staging"
    assert call.kwargs["archive_existing_versions"] is False


def test_tags_include_recall_and_model_name(mock_client):
    mock_client.search_model_versions.return_value = []

    train.register_best_model(_best_run(), model_registry_name="M")

    tag_keys = {
        call.kwargs.get("key") for call in mock_client.set_model_version_tag.call_args_list
    }
    assert {"model_name", "test_recall", "cv_mean"}.issubset(tag_keys)


def test_registry_creation_failure_is_non_fatal(mock_client):
    """An existing registered model raises on create; registration must continue."""
    mock_client.create_registered_model.side_effect = Exception("already exists")
    mock_client.search_model_versions.return_value = []

    version = train.register_best_model(_best_run(), model_registry_name="M")

    assert version == "1"
    mock_client.create_model_version.assert_called_once()
