import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class MemoryItem(BaseModel):
    """
    Define memory item
    """
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}
    
    # Additional fields for working memory
    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """
    Define configuration parameters for memory system, including storage, statistics, and specific settings for different memory types
    """
    # Storage path
    storage_path: str = "./memory_data"
    
    # Basic configuration for statistics display (for display only)
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # Working memory specific configuration
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # Perceptual memory specific configuration
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """
    Define base class for memory system - All memory types should inherit from this class and implement the abstract methods
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")


    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        Add Memory

        Args:
            memory_item: Memory item object

        Returns:
            Memory ID
        """
        pass


    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """
        Retrieve Memories based on query and optional filters

        Args:
            query: Query content
            limit: Return quantity limit
            **kwargs: Other retrieval parameters

        Returns:
            List of related memories
        """
        pass


    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update Memory

        Args:
            memory_id: Memory ID
            content: New content
            importance: New importance
            metadata: New metadata

        Returns:
            Whether update was successful
        """
        pass


    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """Delete Memory

        Args:
            memory_id: Memory ID

        Returns:
            Whether deletion was successful
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """
        Check if memory exists

        Args:
            memory_id: Memory ID
            
        Returns:
            Whether memory exists
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all memories
        """
        pass


    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics, such as total count, average importance, etc.

        Returns:
            Dictionary of memory statistics
        """
        pass


    def _generate_id(self) -> str:
        """
        Generate unique ID for memory item
        """
        return str(uuid.uuid4())


    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """
        Calculate importance score for a memory item based on its content and other factors

        Args:
            content: Memory content
            base_importance: Base importance score (default 0.5)        

        Returns:
            Calculated importance score (0.0 to 1.0)
        """
        importance = base_importance

        # Based on content length
        if len(content) > 100:
            importance += 0.1

        # Based on keywords
        important_keywords = ["important", "key", "must", "note", "warning", "error"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))


    def __str__(self) -> str:
        """
        String representation of the memory system, showing its type and basic statistics
        """
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"


    def __repr__(self) -> str:
        """
        Official string representation of the memory system
        """
        return self.__str__()
