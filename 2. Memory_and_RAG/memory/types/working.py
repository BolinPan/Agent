"""Working Memory Implementation

According to Chapter 8 architecture design for working memory, provides:
- Short-term context management
- Capacity and time limits
- Priority management
- Automatic cleanup mechanism
"""

import heapq
from typing import List, Dict, Any
from datetime import datetime, timedelta
from memory.base import BaseMemory, MemoryItem, MemoryConfig


class WorkingMemory(BaseMemory):
    """
    Working Memory Implementation - designed for short-term context management with limited capacity and strong timeliness, suitable for session-level memory storage and retrieval.
    
    Features:
    - Short-term context management: Store recent interactions, events, and information relevant to the current session or task.
    - Capacity and time limits: Implement limits on the number of memory items and their lifespan to ensure that the working memory remains focused and efficient.
    - Priority management: Assign importance scores to memory items and prioritize retrieval based on relevance and importance.
    - Automatic cleanup mechanism: Regularly remove outdated or low-importance memories to free up space and maintain performance.
    """
    

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # Working memory specific configuration
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens

        # Pure memory TTL (minutes), can be overridden by mounting working_memory_ttl_minutes on MemoryConfig
        self.max_age_minutes = getattr(self.config, 'working_memory_ttl_minutes', 120)
        self.current_tokens = 0
        self.session_start = datetime.now()
        
        # Memory storage (working memory does not need persistence)
        self.memories: List[MemoryItem] = []
        
        # Use priority queue to manage memories
        self.memory_heap = []  # (priority, timestamp, memory_item)
    

    def add(self, memory_item: MemoryItem) -> str:
        """
        Add memory item to working memory, with priority calculation and capacity management
        """
        # Expire old memories before adding new one (lazy expiration)
        self._expire_old_memories()

        # Calculate importance and priority
        priority = self._calculate_priority(memory_item)
        
        # Add to heap and list
        heapq.heappush(self.memory_heap, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)
        
        # Update token count
        self.current_tokens += len(memory_item.content.split())
        
        # Enforce capacity limits after adding new memory
        self._enforce_capacity_limits()
        
        return memory_item.id
    
    
    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:
        """
        Retrieve relevant memories from working memory based on query, with hybrid relevance scoring and optional user filtering
        """
        # Expire old memories before retrieval (lazy expiration)
        self._expire_old_memories()
        if not self.memories:
            return []

        # Filter forgotten memories
        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]
        
        # Filter by user ID (if provided)
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        # Try semantic vector retrieval (if embedding model available)
        vector_scores = {}
        try:
            # Simple semantic similarity calculation (using TF-IDF or other lightweight method)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Prepare documents
            documents = [query] + [m.content for m in filtered_memories]
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarity
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Store vector scores
            for i, memory in enumerate(filtered_memories):
                vector_scores[memory.id] = similarities[i]
                
        except Exception as e:
            # If vector retrieval fails, fallback to keyword matching
            vector_scores = {}

        # Calculate final scores
        query_lower = query.lower()
        scored_memories = []
        
        for memory in filtered_memories:
            content_lower = memory.content.lower()
            
            # Get vector score (if available)
            vector_score = vector_scores.get(memory.id, 0.0)
            
            # Keyword matching score
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # Word matching
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8

            # Hybrid score: vector retrieval + keyword matching
            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score
            
            # Time decay
            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay
            
            # Importance weight
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight
            
            if final_score > 0:
                scored_memories.append((final_score, memory))

        # Sort by score and return
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Update working memory"""
        for memory in self.memories:
            if memory.id == memory_id:
                old_tokens = len(memory.content.split())
                
                if content is not None:
                    memory.content = content
                    # Update token count
                    new_tokens = len(content.split())
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens
                
                if importance is not None:
                    memory.importance = importance
                
                if metadata is not None:
                    memory.metadata.update(metadata)
                
                # Recalculate priority and update heap
                self._update_heap_priority(memory)
                
                return True
        return False
    

    def remove(self, memory_id: str) -> bool:
        """
        Delete memory from working memory, with heap management and token count update
        """
        # Find memory in list
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                # Delete from list
                removed_memory = self.memories.pop(i)

                # Delete from heap (mark deleted)
                self._mark_deleted_in_heap(memory_id)
                
                # Update token count
                self.current_tokens -= len(removed_memory.content.split())
                self.current_tokens = max(0, self.current_tokens)
                
                return True
        return False
    

    def has_memory(self, memory_id: str) -> bool:
        """
        Check if memory exists in working memory
        """
        return any(memory.id == memory_id for memory in self.memories)
    

    def clear(self):
        """
        Clear all memories from working memory, reset token count and heap
        """
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        # Expire cleanup (lazy)
        self._expire_old_memories()
        
        # Memories in working memory are all active (forgotten memories are directly deleted)
        active_memories = self.memories
        
        return {
            "count": len(active_memories),  # Active memory count
            "forgotten_count": 0,  # Forgotten memories in working memory are directly deleted
            "total_count": len(self.memories),  # Total memory count
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }
    
    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """Get recent memories"""
        sorted_memories = sorted(
            self.memories, 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        return sorted_memories[:limit]
    
    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """Get important memories"""
        sorted_memories = sorted(
            self.memories,
            key=lambda x: x.importance,
            reverse=True
        )
        return sorted_memories[:limit]

    def get_all(self) -> List[MemoryItem]:
        """Get all memories"""
        return self.memories.copy()
    
    def get_context_summary(self, max_length: int = 500) -> str:
        """Get context summary"""
        if not self.memories:
            return "No working memories available."
        
        # Sort by importance and time
        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        summary_parts = []
        current_length = 0
        
        for memory in sorted_memories:
            content = memory.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                # Truncate last memory
                remaining = max_length - current_length
                if remaining > 50:  # Keep at least 50 characters
                    summary_parts.append(content[:remaining] + "...")
                break
        
        return "Working Memory Context:\n" + "\n".join(summary_parts)
    
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 1) -> int:
        """Working memory forgetting mechanism"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove = []
        
        # Always execute TTL expiration first (minute level)
        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)
        
        if strategy == "importance_based":
            # Delete low importance memories
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        
        elif strategy == "time_based":
            # Delete expired memories (working memory usually calculated in hours)
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)
        
        elif strategy == "capacity_based":
            # Delete memories exceeding capacity
            if len(self.memories) > self.max_capacity:
                # Sort by priority, delete lowest
                sorted_memories = sorted(
                    self.memories,
                    key=lambda m: self._calculate_priority(m)
                )
                excess_count = len(self.memories) - self.max_capacity
                for memory in sorted_memories[:excess_count]:
                    to_remove.append(memory.id)
        
        # Execute deletion
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
        
        return forgotten_count
    

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """
        Calculate priority score for a memory item based on importance and time decay
        """
        # Base priority on importance
        priority = memory.importance
        
        # Time decay
        time_decay = self._calculate_time_decay(memory.timestamp)
        priority *= time_decay
        return priority
    

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """
        Calculate time decay factor for a memory item based on its age
        """
        # Calculate hours passed since memory was created
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        
        # Exponential decay (working memory decays faster)
        decay_factor = self.config.decay_factor ** (hours_passed / 6)  # Decay every 6 hours
        return max(0.1, decay_factor)  # Keep minimum 10% weight
    

    def _enforce_capacity_limits(self):
        """
        Enforce capacity and token limits by removing low priority memories if necessary
        """
        # Check memory count limit
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        
        # Check token limit
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()


    def _expire_old_memories(self):
        """
        Expire old memories based on TTL
        """
        # If no memories, skip
        if not self.memories:
            return
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        # Keep memories newer than cutoff, remove older ones
        kept: List[MemoryItem] = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp >= cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())

        # If no memories were removed, skip
        if len(kept) == len(self.memories):
            return
        
        # Update memory list and token count
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        
        # Rebuild heap after expiration
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))
    

    def _remove_lowest_priority_memory(self):
        """
        Remove lowest priority memory to free up space
        """
        # If no memories, skip
        if not self.memories:
            return
        
        # Find memory with lowest priority
        lowest_priority = float('inf')
        lowest_memory = None
        
        # Since we maintain a heap, the lowest priority memory should be at the end of the heap
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory

        # Remove lowest priority memory
        if lowest_memory:
            self.remove(lowest_memory.id)
    
    
    def _update_heap_priority(self, memory: MemoryItem):
        """
        Update memory priority in heap after modification
        """
        # # Since heapq does not support direct priority update, we need to rebuild the heap
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))
    
    
    def _mark_deleted_in_heap(self, memory_id: str):
        """Mark deleted memory in heap"""
        # Since heapq does not support direct deletion, we mark as deleted
        # Will be cleaned up in subsequent operations
        pass
