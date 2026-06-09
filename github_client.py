"""GitHub REST API client for repository discovery and enrichment."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import requests


class GitHubClientError(RuntimeError):
    pass


@dataclass(slots=True)
class GitHubRateLimit:
    remaining: int | None
    reset_at: datetime | None


class GitHubClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout_seconds: float = 20.0,
        max_retries: int = 4,
        sleep_on_rate_limit: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.session = session or requests.Session()
        token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "osiris-repository-ingestion-pipeline",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def search_repositories(
        self,
        query: str,
        *,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 50,
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            response = self._request(
                "GET",
                "/search/repositories",
                params={"q": query, "sort": sort, "order": order, "per_page": per_page, "page": page},
            )
            items = response.get("items") or []
            results.extend(items)
            if len(items) < per_page:
                break
        return results

    def get_repository(self, full_name: str) -> dict[str, Any] | None:
        return self._optional_request("GET", f"/repos/{full_name}")

    def get_readme(self, full_name: str) -> dict[str, Any] | None:
        return self._optional_request("GET", f"/repos/{full_name}/readme")

    def get_topics(self, full_name: str) -> list[str]:
        repo = self._optional_request("GET", f"/repos/{full_name}/topics", headers={"Accept": "application/vnd.github.mercy-preview+json"})
        if not repo:
            return []
        return list(repo.get("names") or [])

    def get_languages(self, full_name: str) -> dict[str, int]:
        return self._optional_request("GET", f"/repos/{full_name}/languages") or {}

    def get_contributors(self, full_name: str, *, per_page: int = 100, max_pages: int = 1) -> list[dict[str, Any]]:
        contributors: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            chunk = self._optional_request(
                "GET",
                f"/repos/{full_name}/contributors",
                params={"anon": "true", "per_page": per_page, "page": page},
            )
            if not chunk:
                break
            contributors.extend(chunk)
            if len(chunk) < per_page:
                break
        return contributors

    def get_events(self, full_name: str, *, per_page: int = 100) -> list[dict[str, Any]]:
        return self._optional_request("GET", f"/repos/{full_name}/events", params={"per_page": per_page}) or []

    def get_stargazers(self, full_name: str, *, per_page: int = 100, max_pages: int = 4) -> list[dict[str, Any]]:
        stargazers: list[dict[str, Any]] = []
        headers = {"Accept": "application/vnd.github.star+json"}
        for page in range(1, max_pages + 1):
            chunk = self._optional_request(
                "GET",
                f"/repos/{full_name}/stargazers",
                params={"per_page": per_page, "page": page},
                headers=headers,
            )
            if not chunk:
                break
            stargazers.extend(chunk)
            if len(chunk) < per_page:
                break
        return stargazers

    def iter_search_repositories(self, query: str, *, sort: str, order: str = "desc", per_page: int = 50, max_pages: int = 2) -> Iterator[dict[str, Any]]:
        for repo in self.search_repositories(query, sort=sort, order=order, per_page=per_page, max_pages=max_pages):
            yield repo

    def rate_limit(self) -> GitHubRateLimit:
        data = self._request("GET", "/rate_limit")
        core = (data.get("resources") or {}).get("core") or {}
        reset = core.get("reset")
        return GitHubRateLimit(
            remaining=core.get("remaining"),
            reset_at=datetime.fromtimestamp(reset, tz=timezone.utc) if reset else None,
        )

    def _optional_request(self, method: str, path: str, **kwargs: Any) -> Any | None:
        try:
            return self._request(method, path, **kwargs)
        except GitHubClientError as exc:
            if "404" in str(exc) or "409" in str(exc) or "451" in str(exc):
                return None
            raise

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = dict(kwargs.pop("headers", {}) or {})
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout_seconds,
                    headers=headers or None,
                    **kwargs,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub request failed for {path}: {exc}") from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 403 and self._is_rate_limited(response):
                if attempt >= self.max_retries or not self.sleep_on_rate_limit:
                    raise GitHubClientError(f"GitHub rate limit exceeded for {path}")
                self._sleep_until_reset(response)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub transient failure {response.status_code} for {path}: {response.text[:300]}")
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 60))
                else:
                    self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise GitHubClientError(f"GitHub error {response.status_code} for {path}: {response.text[:300]}")

            if not response.content:
                return None
            return response.json()

        raise GitHubClientError(f"GitHub request exhausted retries for {path}")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        return remaining == "0" or "rate limit" in response.text.lower()

    def _sleep_until_reset(self, response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            sleep_for = max(int(reset) - int(time.time()) + 2, 1)
        else:
            retry_after = response.headers.get("Retry-After")
            sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 60
        time.sleep(min(sleep_for, 300))

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min((2**attempt) + random.random(), 30.0))
