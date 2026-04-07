"""情景记忆实现

按照第8章架构设计的情景记忆，提供：
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import math
import json
import logging

logger = logging.getLogger(__name__)

from ..base import BaseMemory, MemoryItem, MemoryConfig
from storage import SQLiteDocumentStore, QdrantVectorStore
from ..embedding import get_text_embedder, get_dimension

class Episode:
    """情景记忆中的单个情景"""
    
    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance

class EpisodicMemory(BaseMemory):
    """情景记忆实现
    
    特点：
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 本地缓存（内存）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids
        
        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 权威文档存储（SQLite）
        db_dir = self.config.storage_path if hasattr(self.config, 'storage_path') else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 统一嵌入模型（多语言，默认384维）
        self.embedder = get_text_embedder()

        # 向量存储（Qdrant - 使用连接管理器避免重复连接）
        from ..storage.qdrant_store import QdrantConnectionManager
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=get_dimension(getattr(self.embedder, 'dimension', 384)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine")
        )
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])
        
        # 创建情景（内存缓存）
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 1) 权威存储（SQLite）
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags
            }
        )

        # 2) 向量索引（Qdrant）
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[{
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "memory_type": "episodic",
                    "importance": memory_item.importance,
                    "session_id": session_id,
                    "content": memory_item.content
                }],
                ids=[memory_item.id]
            )
        except Exception:
            # 向量入库失败不影响权威存储
            pass

        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """Retrieve episodic memory (structured filtering + semantic vector retrieval)"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold")

        # Structured filtering candidates (from authoritative database)
        candidate_ids: Optional[set] = None
        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            docs = self.doc_store.search_memories(
                user_id=user_id,
                memory_type="episodic",
                start_time=start_ts,
                end_time=end_ts,
                importance_threshold=importance_threshold,
                limit=1000
            )
            candidate_ids = {d["memory_id"] for d in docs}

        # Vector retrieval (Qdrant)
        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()
            where = {"memory_type": "episodic"}
            if user_id:
                where["user_id"] = user_id
            hits = self.vector_store.search_similar(
                query_vector=query_vec,
                limit=max(limit * 5, 20),
                where=where
            )
        except Exception:
            hits = []

        # Filtering and re-ranking
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set()
        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            
            # Check if forgotten
            episode = next((e for e in self.episodes if e.episode_id == mem_id), None)
            if episode and episode.context.get("forgotten", False):
                continue  # Skip forgotten memories
                
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue

            # Read complete record from authoritative database
            doc = self.doc_store.get_memory(mem_id)
            if not doc:
                continue

            # Calculate comprehensive score: vector 0.6 + recency 0.2 + importance 0.2
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))
            
            # New scoring algorithm: vector retrieval purely based on similarity, importance as weighting factor
            # Base relevance score (not affected by importance)
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            
            # Importance as multiplicative weighting factor, range [0.8, 1.2]
            importance_weight = 0.8 + (imp * 0.4)
            
            # Final score: relevance * importance weight
            combined = base_relevance * importance_weight

            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score
                }
            )
            results.append((combined, item))
            seen.add(mem_id)

        # If vector retrieval has no results, fallback to simple keyword matching (memory cache)
        if not results:
            fallback = super()._generate_id  # Placeholder to avoid unused warning
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id, session_id, time_range):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0))
                    # Fallback matching: new scoring algorithm
                    keyword_score = 0.5  # Base score for simple keyword matching
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined
                        }
                    )
                    results.append((combined, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Update episodic memory (SQLite as authoritative, Qdrant re-embedded as needed)"""
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        # Update SQLite
        doc_updated = self.doc_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata
        )

        # If content changes, re-embed and upsert to Qdrant
        if content is not None:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                # Get updated record to sync payload
                doc = self.doc_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "episodic",
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
                    "session_id": (doc.get("properties", {}) or {}).get("session_id"),
                    "content": content
                }
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[payload],
                    ids=[memory_id]
                )
            except Exception:
                pass

        return updated or doc_updated
    
    def remove(self, memory_id: str) -> bool:
        """Delete episodic memory (SQLite + Qdrant)"""
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break

        # Authoritative database deletion
        doc_deleted = self.doc_store.delete_memory(memory_id)
        
        # Vector database deletion
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception:
            pass
        
        return removed or doc_deleted
    
    def has_memory(self, memory_id: str) -> bool:
        """Check if memory exists"""
        return any(episode.episode_id == memory_id for episode in self.episodes)
    
    def clear(self):
        """Clear all episodic memories (only clears episodic, does not affect other types)"""
        # Memory cache
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

        # Delete all episodic in SQLite
        docs = self.doc_store.search_memories(memory_type="episodic", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)

        # Delete corresponding vectors by ID in Qdrant
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception:
            pass

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        """Episodic memory forgetting mechanism (hard deletion)"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove = []  # Collect memories to delete
        
        for episode in self.episodes:
            should_forget = False
            
            if strategy == "importance_based":
                # Forget based on importance
                if episode.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                # Forget based on time
                cutoff_time = current_time - timedelta(days=max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                # Forget based on capacity (keep most important)
                if len(self.episodes) > self.config.max_capacity:
                    sorted_episodes = sorted(self.episodes, key=lambda e: e.importance)
                    excess_count = len(self.episodes) - self.config.max_capacity
                    if episode in sorted_episodes[:excess_count]:
                        should_forget = True
            
            if should_forget:
                to_remove.append(episode.episode_id)
        
        # Execute hard deletion
        for episode_id in to_remove:
            if self.remove(episode_id):
                forgotten_count += 1
                logger.info(f"Episodic memory hard deletion: {episode_id[:8]}... (Strategy: {strategy})")
        
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """Get all episodic memories (converted to MemoryItem format)"""
        memory_items = []
        for episode in self.episodes:
            memory_item = MemoryItem(
                id=episode.episode_id,
                content=episode.content,
                memory_type="episodic",
                user_id=episode.user_id,
                timestamp=episode.timestamp,
                importance=episode.importance,
                metadata=episode.metadata
            )
            memory_items.append(memory_item)
        return memory_items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics (merged from SQLite and Qdrant)"""
        # Hard deletion mode: all episodes are active
        active_episodes = self.episodes
        
        db_stats = self.doc_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {"store_type": "qdrant"}
        return {
            "count": len(active_episodes),  # Active memory count
            "forgotten_count": 0,  # Forgotten memories are directly deleted in hard deletion mode
            "total_count": len(self.episodes),  # Total memory count
            "sessions_count": len(self.sessions),
            "avg_importance": sum(e.importance for e in active_episodes) / len(active_episodes) if active_episodes else 0.0,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }
    
    def get_session_episodes(self, session_id: str) -> List[Episode]:
        """Get all episodes for the specified session"""
        if session_id not in self.sessions:
            return []
        
        episode_ids = self.sessions[session_id]
        return [e for e in self.episodes if e.episode_id in episode_ids]
    
    def find_patterns(self, user_id: str = None, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """Discover user behavior patterns"""
        # Check cache
        cache_key = f"{user_id}_{min_frequency}"
        if (cache_key in self.patterns_cache and 
            self.last_pattern_analysis and 
            (datetime.now() - self.last_pattern_analysis).hours < 1):
            return self.patterns_cache[cache_key]
        
        # Filter episodes
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        
        # Simple pattern recognition: based on content keywords
        keyword_patterns = {}
        context_patterns = {}
        
        for episode in episodes:
            # Extract keywords
            words = episode.content.lower().split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + 1
            
            # Extract context patterns
            for key, value in episode.context.items():
                pattern_key = f"{key}:{value}"
                context_patterns[pattern_key] = context_patterns.get(pattern_key, 0) + 1
        
        # Filter frequent patterns
        patterns = []
        
        for keyword, frequency in keyword_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "keyword",
                    "pattern": keyword,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        
        for context_pattern, frequency in context_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "context",
                    "pattern": context_pattern,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        
        # Sort by frequency
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        
        # Cache results
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()
        
        return patterns
    
    def get_timeline(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get timeline view"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        
        timeline = []
        for episode in episodes[:limit]:
            timeline.append({
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp.isoformat(),
                "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                "session_id": episode.session_id,
                "importance": episode.importance,
                "outcome": episode.outcome
            })
        
        return timeline
    
    def _filter_episodes(
        self,
        user_id: str = None,
        session_id: str = None,
        time_range: Tuple[datetime, datetime] = None
    ) -> List[Episode]:
        """Filter episodes"""
        filtered = self.episodes
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]
        
        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]
        
        return filtered
    
    def _calculate_time_span(self) -> float:
        """Calculate memory time span (days)"""
        if not self.episodes:
            return 0.0
        
        timestamps = [e.timestamp for e in self.episodes]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        return (max_time - min_time).days
    
    def _persist_episode(self, episode: Episode):
        """Persist episode to storage backend"""
        if self.storage and hasattr(self.storage, 'add_memory'):
            self.storage.add_memory(
                memory_id=episode.episode_id,
                user_id=episode.user_id,
                content=episode.content,
                memory_type="episodic",
                timestamp=int(episode.timestamp.timestamp()),
                importance=episode.importance,
                properties={
                    "session_id": episode.session_id,
                    "context": episode.context,
                    "outcome": episode.outcome
                }
            )
    
    def _remove_from_storage(self, memory_id: str):
        """Delete from storage backend"""
        if self.storage and hasattr(self.storage, 'delete_memory'):
            self.storage.delete_memory(memory_id)
