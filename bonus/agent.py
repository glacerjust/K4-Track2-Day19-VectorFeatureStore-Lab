"""HybridMemoryAgent — Kết hợp Episodic Memory (Qdrant) và Profile Features (Feast).

Bonus Challenge — Lab 19 (Vector Store + Feature Store).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.embeddings import Embedder

COLLECTION_NAME = "bonus_episodic_memory"
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class UserProfile:
    user_id: str
    reading_speed_wpm: int = 220
    preferred_language: str = "vi"
    topic_affinity: str = "cloud"
    queries_last_hour: int = 5
    distinct_topics_24h: int = 3


class HybridMemoryAgent:
    """Agent quản lý bộ nhớ lai: Episodic Memory (Qdrant) + Behavioral Profile (Feast)."""

    def __init__(self, in_memory: bool = True) -> None:
        self.embedder = Embedder()
        self.dim = self.embedder.dim
        self.client = QdrantClient(":memory:" if in_memory else "http://localhost:6333")
        self._init_qdrant()
        self._point_id = 0
        self._local_profile_cache: dict[str, UserProfile] = {}

    def _init_qdrant(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def _get_profile_from_feast(self, user_id: str) -> UserProfile:
        """Đọc profile từ Feast online store; fallback về default/cache nếu offline."""
        try:
            from feast import FeatureStore

            feast_dir = ROOT / "app" / "feast_repo"
            if (feast_dir / "registry.db").exists():
                fs = FeatureStore(repo_path=str(feast_dir))
                features = fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                return UserProfile(
                    user_id=user_id,
                    reading_speed_wpm=features.get("reading_speed_wpm", [220])[0] or 220,
                    preferred_language=features.get("preferred_language", ["vi"])[0] or "vi",
                    topic_affinity=features.get("topic_affinity", ["cloud"])[0] or "cloud",
                    queries_last_hour=features.get("queries_last_hour", [5])[0] or 5,
                    distinct_topics_24h=features.get("distinct_topics_24h", [3])[0] or 3,
                )
        except Exception:
            pass

        # Fallback profile thông minh theo từng user
        if user_id not in self._local_profile_cache:
            self._local_profile_cache[user_id] = UserProfile(
                user_id=user_id,
                reading_speed_wpm=240,
                preferred_language="vi",
                topic_affinity="cloud",
                queries_last_hour=8,
                distinct_topics_24h=4,
            )
        return self._local_profile_cache[user_id]

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Lưu một đoạn ký ức mới vào Episodic Memory của người dùng."""
        text_clean = text.strip()
        if not text_clean:
            return

        vector = next(self.embedder.embed([text_clean])).tolist()
        point = PointStruct(
            id=self._point_id,
            vector=vector,
            payload={
                "user_id": user_id,
                "text": text_clean,
                "timestamp": time.time(),
            },
        )
        self._point_id += 1
        self.client.upsert(collection_name=COLLECTION_NAME, points=[point])

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Trích xuất ký ức liên quan + đặc trưng người dùng và lắp ghép context hoàn chỉnh."""
        # 1. Lấy profile tĩnh và hành vi gần đây từ Feature Store
        profile = self._get_profile_from_feast(user_id)

        # 2. Truy vấn vector từ Episodic Memory có lọc theo đúng user_id
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k,
        ).points

        # 3. Lắp ghép ngữ cảnh (Context Assembly)
        memories = [f"- [{h.score:.3f}] {h.payload['text']}" for h in hits]
        memory_block = "\n".join(memories) if memories else "- (Chưa có ký ức nào liên quan)"

        context = (
            f"=== [ASSEMBLED AGENT CONTEXT] ===\n"
            f"[User Profile] ID: {profile.user_id} | Ngôn ngữ ưu tiên: {profile.preferred_language.upper()} "
            f"| Tốc độ đọc: {profile.reading_speed_wpm} wpm | Chủ đề yêu thích: {profile.topic_affinity}\n"
            f"[Recent Activity] Tần suất: {profile.queries_last_hour} queries/giờ qua "
            f"| Số chủ đề đã xem: {profile.distinct_topics_24h} topics/24h\n"
            f"[Episodic Memories Retrieved (Top {len(hits)})]:\n"
            f"{memory_block}\n"
            f"=== [READY FOR PROMPT GENERATION] ==="
        )
        return context
