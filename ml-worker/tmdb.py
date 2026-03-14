"""
TMDB API client for fetching cast/character information.
Fetches episode-level cast for TV shows, general cast for movies.
"""

import json
import logging
import urllib.request
import urllib.parse

logger = logging.getLogger("subarr-worker")

TMDB_BASE = "https://api.themoviedb.org/3"


class TMDBClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        params = params or {}
        params["api_key"] = self.api_key
        url = f"{TMDB_BASE}{path}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error(f"TMDB request failed: {e}")
            return None

    def get_characters(self, metadata: dict) -> list[dict]:
        """
        Get character list from TMDB based on metadata.

        For TV: uses season/episode for episode-specific cast.
        For movies: uses general credits.
        Falls back to search by title if tmdb_id not provided.

        Returns: [{"actor": "...", "character": "..."}, ...]
        """
        title = metadata.get("title")
        tmdb_id = metadata.get("tmdb_id")
        season = metadata.get("season")
        episode = metadata.get("episode")
        media_type = metadata.get("media_type")

        # If no tmdb_id, search by title
        if not tmdb_id and title:
            tmdb_id, media_type = self._search(title)

        if not tmdb_id:
            logger.warning("No TMDB ID found, skipping character fetch")
            return []

        # TV show with episode info → episode cast, enrich with season cast
        if media_type == "tv" and season and episode:
            characters = self._get_episode_credits(tmdb_id, season, episode)
            if characters:
                # Enrich with season cast for minor characters not in episode credits
                season_chars = self._get_season_credits(tmdb_id, season)
                if season_chars:
                    existing = {c["character"] for c in characters}
                    for sc in season_chars:
                        if sc["character"] not in existing:
                            characters.append(sc)
                    logger.info(f"Enriched episode cast ({len(characters)} total) with season credits")
                return characters
            # Fallback to season credits
            logger.info("Episode credits empty, falling back to season credits")

        # TV show without episode → season or general cast
        if media_type == "tv":
            if season:
                characters = self._get_season_credits(tmdb_id, season)
                if characters:
                    return characters
            return self._get_tv_credits(tmdb_id)

        # Movie
        return self._get_movie_credits(tmdb_id)

    def _search(self, query: str) -> tuple[int | None, str | None]:
        data = self._get("/search/multi", {"query": query})
        if not data:
            return None, None
        results = data.get("results", [])
        if not results:
            logger.warning(f"TMDB search '{query}' returned no results")
            return None, None
        result = results[0]
        media_type = result.get("media_type", "movie")
        name = result.get("title") or result.get("name")
        logger.info(f"TMDB found: {name} (id={result['id']}, type={media_type})")
        return result["id"], media_type

    def _get_episode_credits(self, tv_id: int, season: int, episode: int) -> list[dict]:
        data = self._get(f"/tv/{tv_id}/season/{season}/episode/{episode}/credits")
        if not data:
            return []
        return self._extract_cast(data.get("cast", []))

    def _get_season_credits(self, tv_id: int, season: int) -> list[dict]:
        data = self._get(f"/tv/{tv_id}/season/{season}/aggregate_credits")
        if not data:
            return []
        return self._extract_aggregate_cast(data.get("cast", []))

    def _get_tv_credits(self, tv_id: int) -> list[dict]:
        data = self._get(f"/tv/{tv_id}/aggregate_credits")
        if not data:
            return []
        return self._extract_aggregate_cast(data.get("cast", []))

    def _get_movie_credits(self, movie_id: int) -> list[dict]:
        data = self._get(f"/movie/{movie_id}/credits")
        if not data:
            return []
        return self._extract_cast(data.get("cast", []))

    @staticmethod
    def _extract_cast(cast: list, limit: int = 20) -> list[dict]:
        return [
            {"actor": m.get("name", ""), "character": m.get("character", "Unknown")}
            for m in cast[:limit]
            if m.get("character")
        ]

    @staticmethod
    def _extract_aggregate_cast(cast: list, limit: int = 20) -> list[dict]:
        result = []
        for m in cast[:limit]:
            roles = m.get("roles", [])
            character = roles[0]["character"] if roles else "Unknown"
            if character:
                result.append({"actor": m.get("name", ""), "character": character})
        return result
