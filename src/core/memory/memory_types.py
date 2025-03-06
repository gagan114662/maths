"""
Memory types for the memory management system.
"""
import enum
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class MemoryType(str, enum.Enum):
    """Types of memories stored in the system."""
    MARKET_DATA = "market_data"
    STRATEGY = "strategy"
    BACKTEST = "backtest"
    RISK = "risk"
    AGENT_INTERACTION = "agent_interaction"
    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"
    RESEARCH = "research"
    DECISION = "decision"


class MemoryImportance(int, enum.Enum):
    """Importance levels for memory items."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Memory(BaseModel):
    """
    A memory item stored in the memory system.
    
    Attributes:
        type: The type of memory
        content: The memory content
        metadata: Additional metadata about the memory
        importance: The importance level of the memory
        created_at: When the memory was created
        accessed_at: When the memory was last accessed
        access_count: How many times the memory has been accessed
        tags: Tags for categorizing the memory
    """
    type: MemoryType
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    tags: List[str] = Field(default_factory=list)
    
    def update_content(self, new_content: Dict[str, Any]) -> 'Memory':
        """Update the memory content."""
        self.content.update(new_content)
        return self
    
    def mark_accessed(self) -> 'Memory':
        """Mark the memory as accessed."""
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self
    
    def add_tag(self, tag: str) -> 'Memory':
        """Add a tag to the memory."""
        if tag not in self.tags:
            self.tags.append(tag)
        return self
    
    def remove_tag(self, tag: str) -> 'Memory':
        """Remove a tag from the memory."""
        if tag in self.tags:
            self.tags.remove(tag)
        return self