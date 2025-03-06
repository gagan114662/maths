"""
Simple memory management system that doesn't require SQLAlchemy.
Uses JSON files for storage instead of a database.
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import shutil

# Import enums directly to avoid circular imports
from .memory.memory_types import MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

class SimpleMemoryManager:
    """
    A simple memory manager that stores entries as JSON files.
    
    This implementation avoids the complexity of SQLAlchemy and
    provides a straightforward way to persist memory entries.
    """
    
    def set_working_memory(self, context_id: str, user_id: Optional[str] = None) -> None:
        """Compatibility method for the original MemoryManager interface."""
        # This is a no-op in the simple implementation, but provides API compatibility
        pass
    
    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize the simple memory manager.
        
        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir)
        self.entries_dir = self.memory_dir / "entries"
        self.index_path = self.memory_dir / "index.json"
        self.cache = {}
        self.cache_size = 1000
        
        # Create directories
        self._create_dirs()
        
        # Load or create index
        self._load_index()
    
    def _create_dirs(self):
        """Create required directories."""
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        self.entries_dir.mkdir(exist_ok=True)
    
    def _load_index(self):
        """Load or create the memory index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory index: {str(e)}")
                self.index = {"entries": [], "next_id": 1}
        else:
            self.index = {"entries": [], "next_id": 1}
            self._save_index()
    
    def _save_index(self):
        """Save the memory index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_entry_path(self, memory_id: int) -> Path:
        """Get the file path for a memory entry."""
        return self.entries_dir / f"{memory_id}.json"
    
    def store(
        self,
        memory_type: Union[str, MemoryType],
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None,
        importance: Optional[Union[str, MemoryImportance]] = None,  # Added for compatibility
        context_id: Optional[str] = None,  # Added for compatibility
        ref_id: Optional[str] = None,      # Added for compatibility
        tags: Optional[List[str]] = None   # Added for compatibility
    ) -> int:
        """
        Store a memory entry.
        
        Args:
            memory_type: Type of memory entry
            content: Content to store
            metadata: Optional metadata
            vectors: Optional vector embeddings
            importance: Optional importance level
            context_id: Optional context ID
            ref_id: Optional reference ID
            tags: Optional tags
            
        Returns:
            int: Memory entry ID
        """
        # Convert enum to string if enum types provided
        if hasattr(memory_type, 'value'):
            memory_type = memory_type.value
            
        if importance is not None and hasattr(importance, 'value'):
            importance = importance.value
            
        # Update metadata with additional fields
        full_metadata = metadata or {}
        if importance is not None:
            full_metadata['importance'] = importance
        if context_id is not None:
            full_metadata['context_id'] = context_id
        if ref_id is not None:
            full_metadata['ref_id'] = ref_id
        if tags is not None:
            full_metadata['tags'] = tags
        try:
            # Get next ID
            memory_id = self.index["next_id"]
            self.index["next_id"] += 1
            
            # Create entry
            timestamp = datetime.now().isoformat()
            entry = {
                "id": memory_id,
                "type": memory_type,
                "timestamp": timestamp,
                "content": content,
                "metadata": full_metadata,
                "vectors": vectors or {}
            }
            
            # Save to file
            entry_path = self._get_entry_path(memory_id)
            with open(entry_path, 'w') as f:
                json.dump(entry, f, indent=2)
            
            # Update index
            index_entry = {
                "id": memory_id,
                "type": memory_type,
                "timestamp": timestamp
            }
            self.index["entries"].append(index_entry)
            self._save_index()
            
            # Update cache
            self.cache[memory_id] = entry
            self._trim_cache()
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def retrieve(
        self,
        memory_id: Optional[int] = None,
        include_vectors: bool = False,
        memory_type: Optional[Union[str, MemoryType]] = None,
        tags: Optional[List[str]] = None,
        importance_minimum: Optional[Union[str, MemoryImportance]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Union[Optional[Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]:
        """
        Retrieve a memory entry by ID or search by criteria.
        
        Args:
            memory_id: Memory entry ID (if retrieving a specific entry)
            include_vectors: Whether to include vectors
            memory_type: Filter by memory type
            tags: Filter by tags
            importance_minimum: Minimum importance level
            metadata_filter: Filter by metadata fields
            
        Returns:
            Memory entry, None if not found, or list of entries if searching by criteria
        """
        # Convert enum to string if enum types provided
        if memory_type is not None and hasattr(memory_type, 'value'):
            memory_type = memory_type.value
            
        if importance_minimum is not None and hasattr(importance_minimum, 'value'):
            importance_minimum = importance_minimum.value
        try:
            # If memory_id is provided, retrieve specific entry
            if memory_id is not None:
                # Check cache first
                if memory_id in self.cache:
                    entry = self.cache[memory_id]
                    if not include_vectors:
                        entry = entry.copy()
                        entry.pop("vectors", None)
                    return entry
                
                # Load from file
                entry_path = self._get_entry_path(memory_id)
                if not entry_path.exists():
                    return None
                    
                with open(entry_path, 'r') as f:
                    entry = json.load(f)
                
                # Update cache
                self.cache[memory_id] = entry
                self._trim_cache()
                
                if not include_vectors:
                    entry = entry.copy()
                    entry.pop("vectors", None)
                    
                return entry
            
            # Otherwise, search by criteria (similar to search method)
            else:
                results = []
                for idx_entry in self.index["entries"]:
                    # Type filter
                    if memory_type and idx_entry["type"] != memory_type:
                        continue
                    
                    # Load full entry to check other criteria
                    entry_id = idx_entry["id"]
                    entry_path = self._get_entry_path(entry_id)
                    if not entry_path.exists():
                        continue
                        
                    with open(entry_path, 'r') as f:
                        entry = json.load(f)
                    
                    # Tags filter (assuming tags are in metadata)
                    if tags and not all(tag in entry.get("metadata", {}).get("tags", []) for tag in tags):
                        continue
                    
                    # Importance filter (assuming importance is in metadata)
                    if importance_minimum and entry.get("metadata", {}).get("importance", "") < importance_minimum:
                        continue
                    
                    # Metadata filter
                    if metadata_filter:
                        match = True
                        for key, value in metadata_filter.items():
                            if key not in entry.get("metadata", {}) or entry["metadata"][key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    # Add to results
                    if not include_vectors:
                        entry = entry.copy()
                        entry.pop("vectors", None)
                    
                    results.append((entry_id, entry))
                
                # Sort by timestamp (newest first)
                results.sort(key=lambda x: x[1]["timestamp"], reverse=True)
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return [] if memory_id is None else None
    
    def search(
        self,
        memory_type: Optional[Union[str, MemoryType]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memory entries (simple implementation).
        
        Args:
            memory_type: Optional type filter
            metadata_filter: Optional metadata filter
            top_k: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        # Convert enum to string if enum types provided
        if memory_type is not None and hasattr(memory_type, 'value'):
            memory_type = memory_type.value
            
        try:
            # Filter entries in index
            results = []
            
            for idx_entry in self.index["entries"]:
                # Type filter
                if memory_type and idx_entry["type"] != memory_type:
                    continue
                
                # We need to load the full entry for metadata filtering
                if metadata_filter:
                    entry = self.retrieve(idx_entry["id"])
                    if not entry:
                        continue
                        
                    # Apply metadata filter
                    match = True
                    for key, value in metadata_filter.items():
                        if key not in entry["metadata"] or entry["metadata"][key] != value:
                            match = False
                            break
                            
                    if not match:
                        continue
                        
                    results.append(entry)
                else:
                    # No metadata filter, just add the entry ID
                    results.append(self.retrieve(idx_entry["id"]))
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Return top k
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []
    
    def update(
        self,
        memory_id: int,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None
    ) -> bool:
        """
        Update a memory entry.
        
        Args:
            memory_id: Memory entry ID
            content: Optional new content
            metadata: Optional new metadata
            vectors: Optional new vectors
            
        Returns:
            bool: Success status
        """
        try:
            # Load existing entry
            entry = self.retrieve(memory_id, include_vectors=True)
            if not entry:
                return False
            
            # Update fields
            if content is not None:
                entry["content"] = content
                
            if metadata is not None:
                entry["metadata"].update(metadata)
                
            if vectors is not None:
                entry["vectors"] = vectors
            
            # Save back to file
            entry_path = self._get_entry_path(memory_id)
            with open(entry_path, 'w') as f:
                json.dump(entry, f, indent=2)
            
            # Update cache
            self.cache[memory_id] = entry
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            return False
    
    def delete(self, memory_id: int) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: Memory entry ID
            
        Returns:
            bool: Success status
        """
        try:
            # Check if entry exists
            entry_path = self._get_entry_path(memory_id)
            if not entry_path.exists():
                return False
            
            # Delete file
            entry_path.unlink()
            
            # Update index
            self.index["entries"] = [e for e in self.index["entries"] if e["id"] != memory_id]
            self._save_index()
            
            # Remove from cache
            self.cache.pop(memory_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all memory entries.
        
        Returns:
            bool: Success status
        """
        try:
            # Delete all entry files
            shutil.rmtree(self.entries_dir)
            self.entries_dir.mkdir(exist_ok=True)
            
            # Reset index
            self.index = {"entries": [], "next_id": 1}
            self._save_index()
            
            # Clear cache
            self.cache = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def _trim_cache(self):
        """Trim cache to maximum size."""
        if len(self.cache) > self.cache_size:
            # Remove oldest entries (by timestamp)
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"],
                reverse=False
            )
            
            # Remove oldest entries
            excess = len(self.cache) - self.cache_size
            for k in sorted_keys[:excess]:
                self.cache.pop(k, None)