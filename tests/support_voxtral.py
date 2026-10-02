"""Shared helpers for the test_voxtral test modules (issue #70).

Tests for VoxtralTranscriber.
"""

from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

_EXC_REAL_BY_NAME: dict[str, type] = {
    "GatedRepoError": GatedRepoError,
    "HfHubHTTPError": HfHubHTTPError,
    "RepositoryNotFoundError": RepositoryNotFoundError,
}
