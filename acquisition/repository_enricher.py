"""Transform GitHub repository data into Osiris-compatible payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import logging

from .github_client import GitHubClient
from .github_graphql_client import GitHubGraphQLClient
from utils.readme_processor import ReadmeDocument, process_readme_payload

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class EnrichmentResult:
    repo_id: str
    payload: dict[str, Any]
    raw_repository: dict[str, Any]
    readme: ReadmeDocument
    topics: list[str]
    languages: dict[str, int]


class RepositoryEnricher:
    def __init__(self, client: GitHubClient, graphql_client: GitHubGraphQLClient | None = None) -> None:
        self.client = client
        token = None
        auth_header = getattr(client, "session", None) and getattr(client.session, "headers", {}).get("Authorization")
        if auth_header and isinstance(auth_header, str) and auth_header.startswith("Bearer "):
            token = auth_header[7:]
        self.graphql_client = graphql_client or GitHubGraphQLClient(token=token)

    def enrich(self, repository: dict[str, Any] | str) -> EnrichmentResult | None:
        full_name = repository if isinstance(repository, str) else repository.get("full_name")
        if not full_name:
            return None
        
        # Determine discovery metadata to carry over
        discovery_category = None
        discovery_band = None
        if isinstance(repository, dict):
            discovery_category = repository.get("_discovery_category")
            discovery_band = repository.get("_discovery_band")

        owner, _, name = full_name.partition("/")
        if owner and name:
            try:
                result = self._enrich_graphql(owner, name, discovery_category, discovery_band)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"GraphQL enrichment failed for {full_name}: {e}. Falling back to REST.")

        return self._enrich_rest(full_name, repository)

    def get_repositories_batch(self, repositories: list[dict[str, Any] | str]) -> list[EnrichmentResult]:
        """Fetch multiple repositories using GraphQL batching, falling back to sequential REST if needed."""
        targets = []
        repo_metadata = {}
        
        for repository in repositories:
            full_name = repository if isinstance(repository, str) else repository.get("full_name")
            if not full_name:
                continue
            owner, _, name = full_name.partition("/")
            if owner and name:
                targets.append((owner, name))
                if isinstance(repository, dict):
                    repo_metadata[full_name] = {
                        "_discovery_category": repository.get("_discovery_category"),
                        "_discovery_band": repository.get("_discovery_band")
                    }
        
        results = []
        try:
            # Batch size is usually 10-20 to avoid complexity limit
            for i in range(0, len(targets), 10):
                batch = targets[i:i+10]
                batch_res = self.graphql_client.get_repositories_batch(batch)
                
                for full_name, data in batch_res.items():
                    if not data:
                        continue
                    meta = repo_metadata.get(full_name, {})
                    result = self._process_graphql_data(
                        data, 
                        meta.get("_discovery_category"), 
                        meta.get("_discovery_band")
                    )
                    if result:
                        results.append(result)
                        
        except Exception as e:
            logger.warning(f"GraphQL batch enrichment failed: {e}. Falling back to sequential REST.")
            for repo in repositories:
                res = self.enrich(repo)
                if res:
                    results.append(res)
                    
        return results

    def _enrich_graphql(self, owner: str, name: str, discovery_category: str | None, discovery_band: str | None) -> EnrichmentResult | None:
        data = self.graphql_client.get_repository(owner, name)
        if not data:
            return None
        return self._process_graphql_data(data, discovery_category, discovery_band)

    def _process_graphql_data(self, data: dict[str, Any], discovery_category: str | None, discovery_band: str | None) -> EnrichmentResult | None:
        full_name = data.get("nameWithOwner")
        if not full_name:
            return None
            
        # Reconstruct topics
        topics = []
        topic_nodes = data.get("repositoryTopics", {}).get("nodes", [])
        for node in topic_nodes:
            if "topic" in node and "name" in node["topic"]:
                topics.append(node["topic"]["name"])

        # Reconstruct languages
        languages = {}
        lang_edges = data.get("languages", {}).get("edges", [])
        for edge in lang_edges:
            size = edge.get("size", 0)
            lang_name = edge.get("node", {}).get("name")
            if lang_name:
                languages[lang_name] = size

        # Primary language
        primary_language = None
        if languages:
            primary_language = max(languages.items(), key=lambda item: item[1])[0]

        # Extract README
        readme_text = ""
        for key in ["readme1", "readme2", "readme3", "readme4", "readme5"]:
            blob = data.get(key)
            if blob and blob.get("text"):
                readme_text = blob["text"]
                break
        
        # Because GraphQL returns raw markdown, we construct the equivalent of the REST /readme base64 payload
        import base64
        readme_payload = {
            "content": base64.b64encode(readme_text.encode("utf-8")).decode("ascii"),
            "encoding": "base64"
        } if readme_text else None
        
        readme = process_readme_payload(readme_payload)

        # Star history and events approximation
        stargazers = [{"starred_at": edge.get("starredAt")} for edge in data.get("stargazers", {}).get("edges", [])]
        
        # We approximate events with commits to the default branch
        events = []
        commits = data.get("defaultBranchRef", {}).get("target", {}).get("history", {}).get("nodes", [])
        for commit in commits:
            events.append({
                "type": "PushEvent",
                "created_at": commit.get("committedDate")
            })

        # Contributors / mentionable users
        contributors = data.get("mentionableUsers", {}).get("nodes", [])
        if not contributors:
            contributors = []

        # Construct raw_repository (REST equivalent structure for downstream compatibility)
        raw_repository = {
            "full_name": full_name,
            "name": data.get("name"),
            "description": data.get("description"),
            "html_url": data.get("url"),
            "homepage": data.get("homepageUrl"),
            "created_at": data.get("createdAt"),
            "updated_at": data.get("updatedAt"),
            "pushed_at": data.get("pushedAt"),
            "size": 0, # Cannot get size from GraphQL repo directly easily without languages sum
            "stargazers_count": data.get("stargazerCount", 0),
            "watchers_count": data.get("watchers", {}).get("totalCount", 0),
            "language": primary_language,
            "forks_count": data.get("forkCount", 0),
            "open_issues_count": data.get("issues", {}).get("totalCount", 0),
            "owner": {"login": data.get("owner", {}).get("login")} if data.get("owner") else None,
            "_discovery_category": discovery_category,
            "_discovery_band": discovery_band,
        }
        
        # Calculate size from languages sum as fallback
        if languages:
            raw_repository["size"] = sum(languages.values()) // 1024

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


    def _enrich_rest(self, full_name: str, repository: dict[str, Any] | str) -> EnrichmentResult | None:
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
