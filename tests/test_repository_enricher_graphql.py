"""Tests for RepositoryEnricher GraphQL integration."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from acquisition.repository_enricher import RepositoryEnricher
from acquisition.github_client import GitHubClient
from acquisition.github_graphql_client import GitHubGraphQLClient


class DummyRESTClient(GitHubClient):
    def __init__(self):
        super().__init__(token="dummy")
        self.called = False

    def get_repository(self, full_name: str) -> dict[str, Any] | None:
        self.called = True
        return {
            "full_name": full_name,
            "stargazers_count": 100,
            "language": "Python",
            "size": 100,
            "forks_count": 10,
            "open_issues_count": 5,
        }

    def get_readme(self, full_name: str) -> dict[str, Any] | None:
        return {"content": "IyBEdW1teQ==", "encoding": "base64"}

    def get_topics(self, full_name: str) -> list[str]:
        return ["test"]

    def get_languages(self, full_name: str) -> dict[str, int]:
        return {"Python": 100}

    def get_contributors(self, full_name: str, **kwargs) -> list[dict[str, Any]]:
        return [{"login": "user"}]

    def get_events(self, full_name: str, **kwargs) -> list[dict[str, Any]]:
        return []

    def get_stargazers(self, full_name: str, **kwargs) -> list[dict[str, Any]]:
        return []


class DummyGraphQLClient(GitHubGraphQLClient):
    def __init__(self):
        super().__init__(token="dummy")
        self.called = False
        self.should_fail = False

    def get_repository(self, owner: str, name: str) -> dict[str, Any] | None:
        self.called = True
        if self.should_fail:
            raise Exception("GraphQL error")
            
        return {
            "nameWithOwner": f"{owner}/{name}",
            "name": name,
            "description": "A graphql repo",
            "url": f"https://github.com/{owner}/{name}",
            "stargazerCount": 200,
            "forkCount": 20,
            "languages": {
                "edges": [{"size": 1000, "node": {"name": "Python"}}]
            },
            "repositoryTopics": {
                "nodes": [{"topic": {"name": "graphql"}}]
            },
            "readme1": {"text": "# Title\n\nThis is a GraphQL Readme paragraph."},
            "watchers": {"totalCount": 50},
            "issues": {"totalCount": 10},
            "owner": {"login": owner},
        }

    def get_repositories_batch(self, repos: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        results = {}
        for owner, name in repos:
            results[f"{owner}/{name}"] = self.get_repository(owner, name)
        return results


def test_enricher_uses_graphql_first():
    rest_client = DummyRESTClient()
    gql_client = DummyGraphQLClient()
    
    enricher = RepositoryEnricher(rest_client, gql_client)
    result = enricher.enrich("test/repo")
    
    assert result is not None
    assert gql_client.called is True
    assert rest_client.called is False
    assert result.payload["star_count"] == 200
    assert result.payload["topics"] == ["graphql"]
    assert result.payload["primary_language"] == "Python"
    assert result.payload["readme_length"] > 0


def test_enricher_falls_back_to_rest_on_error():
    rest_client = DummyRESTClient()
    gql_client = DummyGraphQLClient()
    gql_client.should_fail = True
    
    enricher = RepositoryEnricher(rest_client, gql_client)
    result = enricher.enrich("test/repo")
    
    assert result is not None
    assert gql_client.called is True
    assert rest_client.called is True
    assert result.payload["star_count"] == 100


def test_enricher_batch_processing():
    rest_client = DummyRESTClient()
    gql_client = DummyGraphQLClient()
    
    enricher = RepositoryEnricher(rest_client, gql_client)
    repos = [{"full_name": "test/repo1"}, "test/repo2"]
    
    results = enricher.get_repositories_batch(repos)
    assert len(results) == 2
    assert results[0].payload["id"] == "test/repo1"
    assert results[1].payload["id"] == "test/repo2"
    assert rest_client.called is False
