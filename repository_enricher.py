"""Transform GitHub repository data into Osiris-compatible payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from github_client import GitHubClient
from readme_processor import ReadmeDocument, process_readme_payload


@dataclass(slots=True)
class EnrichmentResult:
    repo_id: str
    payload: dict[str, Any]
    raw_repository: dict[str, Any]
    readme: ReadmeDocument
    topics: list[str]
    languages: dict[str, int]


class RepositoryEnricher:
    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def enrich(self, repository: dict[str, Any] | str) -> EnrichmentResult | None:
        full_name = repository if isinstance(repository, str) else repository.get("full_name")
        if not full_name:
            return None

        raw_repository = self.client.get_repository(full_name)
        if not raw_repository:
            return None
        if isinstance(repository, dict):
            raw_repository = {
                **raw_repository,
                "_discovery_category": repository.get("_discovery_category"),
                "_discovery_band": repository.get("_discovery_band"),
            }

        readme_payload = self.client.get_readme(full_name)
        readme = process_readme_payload(readme_payload)
        topics = self._topics(full_name, raw_repository)
        languages = self.client.get_languages(full_name)
        contributors = self.client.get_contributors(full_name, max_pages=1)
        events = self.client.get_events(full_name)
        stargazers = self.client.get_stargazers(full_name, max_pages=4)

        payload = self.to_osiris_payload(
            raw_repository,
            readme=readme,
            topics=topics,
            languages=languages,
            contributors=contributors,
            events=events,
            stargazers=stargazers,
        )
        return EnrichmentResult(
            repo_id=payload["id"],
            payload=payload,
            raw_repository=raw_repository,
            readme=readme,
            topics=topics,
            languages=languages,
        )

    def to_osiris_payload(
        self,
        repository: dict[str, Any],
        *,
        readme: ReadmeDocument,
        topics: list[str],
        languages: dict[str, int],
        contributors: list[dict[str, Any]],
        events: list[dict[str, Any]],
        stargazers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        full_name = repository.get("full_name") or repository.get("name") or "unknown/repository"
        size_kb = int(repository.get("size") or 0)
        primary_language = repository.get("language") or self._primary_language(languages)
        pushed_days_ago = self._days_since(repository.get("pushed_at"))
        deltas = self._estimate_star_deltas(repository, stargazers=stargazers, events=events)

        return {
            "id": full_name,
            "star_count": int(repository.get("stargazers_count") or repository.get("watchers_count") or 0),
            "primary_language": primary_language or "Unknown",
            "readme_length": readme.readme_length,
            "readme_to_codebase_ratio": self._readme_to_codebase_ratio(readme.readme_length, size_kb),
            "mentionable_users_count": self._mentionable_users_count(contributors, repository),
            "delta_3d": deltas[3],
            "delta_7d": deltas[7],
            "delta_30d": deltas[30],
            "extracted_paragraphs": readme.extracted_paragraphs,
            "pushed_days_ago": pushed_days_ago,
            "topics": topics,
            "languages": list(languages.keys()),
            "fork_count": int(repository.get("forks_count") or 0),
            "open_issues_count": int(repository.get("open_issues_count") or 0),
            "description": repository.get("description") or "",
            "html_url": repository.get("html_url"),
            "created_at": repository.get("created_at"),
            "updated_at": repository.get("updated_at"),
            "pushed_at": repository.get("pushed_at"),
            "discovery_category": repository.get("_discovery_category"),
            "discovery_band": repository.get("_discovery_band"),
        }

    def _topics(self, full_name: str, repository: dict[str, Any]) -> list[str]:
        topics = repository.get("topics") or []
        if topics:
            return list(topics)
        try:
            return self.client.get_topics(full_name)
        except Exception:
            return []

    @staticmethod
    def _primary_language(languages: dict[str, int]) -> str | None:
        if not languages:
            return None
        return max(languages.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _readme_to_codebase_ratio(readme_length: int, size_kb: int) -> float:
        codebase_bytes = max(size_kb * 1024, 1)
        return round(readme_length / codebase_bytes, 8)

    @staticmethod
    def _mentionable_users_count(contributors: list[dict[str, Any]], repository: dict[str, Any]) -> int:
        if contributors:
            return min(len(contributors), 100)
        return 1 if repository.get("owner") else 0

    def _estimate_star_deltas(
        self,
        repository: dict[str, Any],
        *,
        stargazers: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[int, int]:
        windows = {3: 0, 7: 0, 30: 0}
        now = datetime.now(timezone.utc)
        timestamps = [self._parse_datetime(item.get("starred_at")) for item in stargazers if item.get("starred_at")]
        timestamps = [value for value in timestamps if value]
        if timestamps:
            for days in windows:
                windows[days] = sum(1 for value in timestamps if (now - value).days <= days)
            return windows

        push_events = [event for event in events if event.get("type") in {"PushEvent", "CreateEvent", "PullRequestEvent", "IssuesEvent"}]
        pushed_days_ago = self._days_since(repository.get("pushed_at"))
        stars = int(repository.get("stargazers_count") or 0)
        activity_multiplier = min(len(push_events) / 30.0, 1.0)
        recency_multiplier = 1.0 if pushed_days_ago <= 3 else 0.6 if pushed_days_ago <= 7 else 0.25 if pushed_days_ago <= 30 else 0.05
        baseline_monthly = max(int((stars ** 0.5) * activity_multiplier * recency_multiplier), 0)
        windows[30] = baseline_monthly
        windows[7] = min(windows[30], max(int(baseline_monthly * 0.35), 0))
        windows[3] = min(windows[7], max(int(baseline_monthly * 0.18), 0))
        return windows

    @staticmethod
    def _days_since(value: str | None) -> int:
        parsed = RepositoryEnricher._parse_datetime(value)
        if not parsed:
            return 999
        return max((datetime.now(timezone.utc) - parsed).days, 0)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
