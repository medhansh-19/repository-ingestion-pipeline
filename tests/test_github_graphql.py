"""Tests for GitHub GraphQL client."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from acquisition.github_graphql_client import GitHubGraphQLClient
from acquisition.github_client import GitHubClientError


def test_graphql_client_successful_query():
    session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "repository": {
                "id": "R_kgDO123",
                "name": "test-repo"
            }
        }
    }
    session.post.return_value = mock_response

    client = GitHubGraphQLClient(token="fake_token", session=session)
    repo = client.get_repository("owner", "test-repo")
    
    assert repo is not None
    assert repo["name"] == "test-repo"
    assert repo["id"] == "R_kgDO123"


def test_graphql_client_handles_partial_errors():
    session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": None,
        "errors": [
            {
                "type": "NOT_FOUND",
                "message": "Could not resolve to a Repository with the name 'test-repo'."
            }
        ]
    }
    session.post.return_value = mock_response

    client = GitHubGraphQLClient(token="fake_token", session=session)
    repo = client.get_repository("owner", "test-repo")
    assert repo is None


def test_graphql_client_raises_on_500():
    session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    session.post.return_value = mock_response

    client = GitHubGraphQLClient(token="fake_token", session=session, max_retries=1)

    with pytest.raises(GitHubClientError) as exc:
        client.get_repository("owner", "test-repo")

    assert "transient failure 500" in str(exc.value)


def test_graphql_client_batch_query():
    session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "repo_0": {"name": "repo1"},
            "repo_1": {"name": "repo2"},
            "repo_2": None
        }
    }
    session.post.return_value = mock_response

    client = GitHubGraphQLClient(token="fake_token", session=session)
    repos = client.get_repositories_batch([("owner1", "repo1"), ("owner2", "repo2"), ("owner3", "repo3")])
    
    assert "owner1/repo1" in repos
    assert repos["owner1/repo1"]["name"] == "repo1"
    assert "owner2/repo2" in repos
    assert repos["owner2/repo2"]["name"] == "repo2"
    assert "owner3/repo3" not in repos
