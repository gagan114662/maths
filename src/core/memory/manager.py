"""
Memory manager for storing and retrieving information.
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import os
from pathlib import Path

from .memory_types import Memory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory manager for storing and retrieving information.
    
    This class provides a flexible memory system with separate
    short-term and long-term memory storage, as well as working
    memory for temporary information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Optional configuration for the memory manager
        """
        self.config = config or {}
        
        # Initialize memory storage
        self.short_term_memory: Dict[str, Memory] = {}
        self.long_term_memory: Dict[str, Memory] = {}
        self.working_memory: Dict[str, Any] = {}
        
        # Configure capacity limits
        self.short_term_capacity = self.config.get("short_term_capacity", 100)
        self.long_term_capacity = self.config.get("long_term_retention", 1000)
        self.importance_threshold = self.config.get("importance_threshold", 0.7)
        
        # Set up memory path
        self.memory_path = Path(self.config.get("memory_path", "memory"))
        self.memory_path.mkdir(exist_ok=True, parents=True)
        
        # Load existing memories if available
        self._load_memories()
        
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            memory_type: The type of memory
            content: The memory content
            metadata: Optional metadata about the memory
            importance: The importance level of the memory
            tags: Optional tags for categorizing the memory
            
        Returns:
            The ID of the stored memory
        """
        # Create a new memory
        memory_id = str(uuid.uuid4())
        memory = Memory(
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            tags=tags or []
        )
        
        # Store in short-term memory
        self.short_term_memory[memory_id] = memory
        logger.debug(f"Stored memory {memory_id} of type {memory_type}")
        
        # Check if we need to consolidate short-term memory
        if len(self.short_term_memory) > self.short_term_capacity:
            self._consolidate_short_term_memory()
            
        # Save to disk
        self._save_memory(memory_id, memory)
        
        return memory_id
    
    def retrieve(
        self,
        memory_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        importance_minimum: Optional[MemoryImportance] = None,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Tuple[str, Memory]]:
        """
        Retrieve memories based on various criteria.
        
        Args:
            memory_id: Optional specific memory ID to retrieve
            memory_type: Optional memory type to filter by
            tags: Optional tags to filter by
            time_range: Optional time range to filter by
            importance_minimum: Optional minimum importance level
            query: Optional query to filter content by
            limit: Optional maximum number of results
            
        Returns:
            A list of (memory_id, memory) tuples
        """
        results = []
        
        # If specific memory ID is provided, try to retrieve it directly
        if memory_id:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                memory.mark_accessed()
                results.append((memory_id, memory))
            return results
        
        # Otherwise, search based on criteria
        # Combine short-term and long-term memories for searching
        all_memories = {**self.short_term_memory, **self.long_term_memory}
        
        for mid, memory in all_memories.items():
            # Filter by memory type
            if memory_type and memory.type != memory_type:
                continue
                
            # Filter by tags (all specified tags must be present)
            if tags and not all(tag in memory.tags for tag in tags):
                continue
                
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                if memory.created_at < start_time or memory.created_at > end_time:
                    continue
                    
            # Filter by importance
            if importance_minimum and memory.importance.value < importance_minimum.value:
                continue
                
            # Filter by content query (simple exact match on key-value pairs)
            if query:
                match = True
                for key, value in query.items():
                    if key not in memory.content or memory.content[key] != value:
                        match = False
                        break
                if not match:
                    continue
                    
            # Add to results
            memory.mark_accessed()
            results.append((mid, memory))
            
            # Check limit
            if limit and len(results) >= limit:
                break
                
        logger.debug(f"Retrieved {len(results)} memories matching criteria")
        return results
    
    def update(
        self,
        memory_id: str,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[MemoryImportance] = None,
        tags_to_add: Optional[List[str]] = None,
        tags_to_remove: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The ID of the memory to update
            content: Optional new content to merge with existing content
            metadata: Optional new metadata to merge with existing metadata
            importance: Optional new importance level
            tags_to_add: Optional tags to add
            tags_to_remove: Optional tags to remove
            
        Returns:
            True if the memory was updated, False if it wasn't found
        """
        memory = self._get_memory_by_id(memory_id)
        if not memory:
            logger.warning(f"Memory {memory_id} not found for update")
            return False
            
        # Update content if provided
        if content:
            memory.update_content(content)
            
        # Update metadata if provided
        if metadata:
            memory.metadata.update(metadata)
            
        # Update importance if provided
        if importance:
            memory.importance = importance
            
        # Add tags if provided
        if tags_to_add:
            for tag in tags_to_add:
                memory.add_tag(tag)
                
        # Remove tags if provided
        if tags_to_remove:
            for tag in tags_to_remove:
                memory.remove_tag(tag)
                
        # Mark as accessed
        memory.mark_accessed()
        
        # Save to disk
        self._save_memory(memory_id, memory)
        
        logger.debug(f"Updated memory {memory_id}")
        return True
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if the memory was deleted, False if it wasn't found
        """
        # Check short-term memory
        if memory_id in self.short_term_memory:
            del self.short_term_memory[memory_id]
            logger.debug(f"Deleted memory {memory_id} from short-term memory")
            
            # Remove from disk
            self._delete_memory_file(memory_id)
            return True
            
        # Check long-term memory
        if memory_id in self.long_term_memory:
            del self.long_term_memory[memory_id]
            logger.debug(f"Deleted memory {memory_id} from long-term memory")
            
            # Remove from disk
            self._delete_memory_file(memory_id)
            return True
            
        logger.warning(f"Memory {memory_id} not found for deletion")
        return False
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """
        Set a value in working memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.working_memory[key] = value
        logger.debug(f"Set working memory key: {key}")
    
    def get_working_memory(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from working memory.
        
        Args:
            key: The key to retrieve
            default: Default value if key is not found
            
        Returns:
            The stored value, or the default if not found
        """
        return self.working_memory.get(key, default)
    
    def clear_working_memory(self, key: Optional[str] = None) -> None:
        """
        Clear working memory.
        
        Args:
            key: Optional specific key to clear, or None to clear all
        """
        if key:
            if key in self.working_memory:
                del self.working_memory[key]
                logger.debug(f"Cleared working memory key: {key}")
        else:
            self.working_memory = {}
            logger.debug("Cleared all working memory")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the memory system state.
        
        Returns:
            A dictionary with memory statistics
        """
        # Count memories by type
        type_counts = {}
        for memory_type in MemoryType:
            type_counts[memory_type.value] = 0
            
        for memory in self.short_term_memory.values():
            type_counts[memory.type.value] += 1
            
        for memory in self.long_term_memory.values():
            type_counts[memory.type.value] += 1
            
        # Count memories by importance
        importance_counts = {}
        for importance in MemoryImportance:
            importance_counts[importance.name] = 0
            
        for memory in self.short_term_memory.values():
            importance_counts[memory.importance.name] += 1
            
        for memory in self.long_term_memory.values():
            importance_counts[memory.importance.name] += 1
            
        # Get all tags
        all_tags = set()
        for memory in list(self.short_term_memory.values()) + list(self.long_term_memory.values()):
            all_tags.update(memory.tags)
            
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "working_memory_keys": list(self.working_memory.keys()),
            "memory_types": type_counts,
            "importance_levels": importance_counts,
            "all_tags": sorted(list(all_tags))
        }
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID from either short-term or long-term memory."""
        if memory_id in self.short_term_memory:
            return self.short_term_memory[memory_id]
        elif memory_id in self.long_term_memory:
            return self.long_term_memory[memory_id]
        else:
            return None
    
    def _consolidate_short_term_memory(self) -> None:
        """
        Move memories from short-term to long-term based on importance.
        
        This helps manage memory capacity limits and ensures important
        memories are retained.
        """
        logger.debug("Consolidating short-term memory")
        
        # Sort memories by importance and recency
        memories_to_consolidate = sorted(
            self.short_term_memory.items(),
            key=lambda x: (x[1].importance.value, x[1].accessed_at or x[1].created_at),
            reverse=True
        )
        
        # Keep only the most important memories in short-term memory
        short_term_count = min(self.short_term_capacity, len(memories_to_consolidate))
        
        new_short_term = {}
        for idx, (memory_id, memory) in enumerate(memories_to_consolidate):
            if idx < short_term_count:
                # Keep in short-term memory
                new_short_term[memory_id] = memory
            else:
                # Consider for long-term memory based on importance
                if memory.importance.value >= MemoryImportance.MEDIUM.value:
                    self.long_term_memory[memory_id] = memory
                    logger.debug(f"Moved memory {memory_id} to long-term memory")
                else:
                    # Memory is not important enough to keep
                    self._delete_memory_file(memory_id)
                    logger.debug(f"Discarded low-importance memory {memory_id}")
        
        # Update short-term memory
        self.short_term_memory = new_short_term
        
        # Check if we need to consolidate long-term memory
        if len(self.long_term_memory) > self.long_term_capacity:
            self._consolidate_long_term_memory()
    
    def _consolidate_long_term_memory(self) -> None:
        """
        Consolidate long-term memory to stay within capacity limits.
        
        This removes the least important and least recently accessed memories.
        """
        logger.debug("Consolidating long-term memory")
        
        # Sort memories by importance, access count, and recency
        memories_to_consolidate = sorted(
            self.long_term_memory.items(),
            key=lambda x: (
                x[1].importance.value, 
                x[1].access_count,
                x[1].accessed_at or x[1].created_at
            ),
            reverse=True
        )
        
        # Keep only up to capacity
        new_long_term = {}
        for idx, (memory_id, memory) in enumerate(memories_to_consolidate):
            if idx < self.long_term_capacity:
                # Keep in long-term memory
                new_long_term[memory_id] = memory
            else:
                # Remove memory
                self._delete_memory_file(memory_id)
                logger.debug(f"Removed memory {memory_id} from long-term memory")
        
        # Update long-term memory
        self.long_term_memory = new_long_term
    
    def _save_memory(self, memory_id: str, memory: Memory) -> None:
        """Save a memory to disk."""
        try:
            file_path = self.memory_path / f"{memory_id}.json"
            with open(file_path, 'w') as f:
                f.write(memory.model_dump_json())
        except Exception as e:
            logger.error(f"Error saving memory {memory_id}: {str(e)}")
    
    def _delete_memory_file(self, memory_id: str) -> None:
        """Delete a memory file from disk."""
        try:
            file_path = self.memory_path / f"{memory_id}.json"
            if file_path.exists():
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting memory file {memory_id}: {str(e)}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        try:
            # Get all memory files
            memory_files = list(self.memory_path.glob("*.json"))
            
            if not memory_files:
                logger.debug("No memory files found to load")
                return
                
            logger.debug(f"Loading {len(memory_files)} memory files")
            
            for file_path in memory_files:
                try:
                    memory_id = file_path.stem
                    
                    with open(file_path, 'r') as f:
                        memory_data = json.load(f)
                        
                    memory = Memory.model_validate(memory_data)
                    
                    # Add to appropriate memory store based on importance
                    if memory.importance.value >= MemoryImportance.MEDIUM.value:
                        self.long_term_memory[memory_id] = memory
                    else:
                        self.short_term_memory[memory_id] = memory
                        
                except Exception as e:
                    logger.warning(f"Error loading memory file {file_path}: {str(e)}")
                    
            logger.debug(f"Loaded {len(self.short_term_memory)} short-term memories " +
                        f"and {len(self.long_term_memory)} long-term memories")
                        
            # Consolidate if needed
            if len(self.short_term_memory) > self.short_term_capacity:
                self._consolidate_short_term_memory()
                
            if len(self.long_term_memory) > self.long_term_capacity:
                self._consolidate_long_term_memory()
                
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")