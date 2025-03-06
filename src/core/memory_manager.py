"""
Memory management system for storing and retrieving agent states and learning outcomes.
"""
import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..utils.config import load_config

class MemoryType(str, Enum):
    """Memory type enumeration."""
    STRATEGY = "strategy"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    RESULT = "result"
    EVALUATION = "evaluation"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    
class MemoryImportance(str, Enum):
    """Memory importance enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

logger = logging.getLogger(__name__)
Base = declarative_base()

class MemoryEntry(Base):
    """SQLAlchemy model for memory entries."""
    __tablename__ = 'memory_entries'
    
    id = Column(Integer, primary_key=True)
    memory_type = Column(String)
    timestamp = Column(DateTime)
    content = Column(JSON)
    meta_data = Column(JSON)  # Renamed to avoid conflict with SQLAlchemy
    vectors = Column(JSON)  # For vector embeddings

class MemoryManager:
    """
    Manages persistent storage and retrieval of system memory.
    
    Attributes:
        config: Configuration dictionary
        engine: SQLAlchemy engine
        Session: SQLAlchemy session maker
    """
    
    def __init__(self, config_path: str = None):
        """Initialize memory manager."""
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize database
        db_path = self.config.get('db_path', 'sqlite:///memory.db')
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize cache
        self.cache = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Create memory directories
        self.memory_dir = Path(self.config.get('memory_dir', 'memory'))
        self._create_memory_dirs()
        
    def store(
        self,
        memory_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None
    ) -> int:
        """
        Store memory entry.
        
        Args:
            memory_type: Type of memory entry
            content: Content to store
            metadata: Optional metadata
            vectors: Optional vector embeddings
            
        Returns:
            int: Memory entry ID
        """
        try:
            # Create session
            session = self.Session()
            
            # Create memory entry
            entry = MemoryEntry(
                memory_type=memory_type,
                timestamp=datetime.now(),
                content=content,
                meta_data=metadata or {},
                vectors=vectors or {}
            )
            
            # Add to database
            session.add(entry)
            session.commit()
            
            # Update cache
            self._update_cache(entry)
            
            # Store large objects separately if needed
            if self._should_store_separately(content):
                self._store_large_content(entry.id, content)
                
            return entry.id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
            
    def retrieve(
        self,
        memory_id: int,
        include_vectors: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory entry by ID.
        
        Args:
            memory_id: Memory entry ID
            include_vectors: Whether to include vector embeddings
            
        Returns:
            Optional[Dict]: Memory entry or None if not found
        """
        # Check cache first
        if memory_id in self.cache:
            return self._prepare_output(self.cache[memory_id], include_vectors)
            
        try:
            # Create session
            session = self.Session()
            
            # Retrieve entry
            entry = session.query(MemoryEntry).filter_by(id=memory_id).first()
            
            if not entry:
                return None
                
            # Check for separate storage
            if self._has_separate_storage(entry.id):
                entry.content.update(self._load_large_content(entry.id))
                
            # Update cache
            self._update_cache(entry)
            
            return self._prepare_output(entry, include_vectors)
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return None
        finally:
            session.close()
            
    def search(
        self,
        memory_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_query: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memory entries.
        
        Args:
            memory_type: Optional type filter
            metadata_filter: Optional metadata filter
            vector_query: Optional vector for similarity search
            top_k: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        try:
            # Create session
            session = self.Session()
            
            # Build query
            query = session.query(MemoryEntry)
            
            if memory_type:
                query = query.filter_by(memory_type=memory_type)
                
            if metadata_filter:
                for key, value in metadata_filter.items():
                    query = query.filter(MemoryEntry.meta_data[key].astext == str(value))
                    
            # Get results
            results = query.all()
            
            # Vector similarity search if needed
            if vector_query is not None:
                results = self._vector_search(results, vector_query, top_k)
            else:
                results = results[:top_k]
                
            return [self._prepare_output(entry) for entry in results]
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []
        finally:
            session.close()
            
    def update(
        self,
        memory_id: int,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None
    ) -> bool:
        """
        Update memory entry.
        
        Args:
            memory_id: Memory entry ID
            content: Optional new content
            metadata: Optional new metadata
            vectors: Optional new vectors
            
        Returns:
            bool: Success status
        """
        try:
            # Create session
            session = self.Session()
            
            # Get entry
            entry = session.query(MemoryEntry).filter_by(id=memory_id).first()
            
            if not entry:
                return False
                
            # Update fields
            if content is not None:
                entry.content = content
                if self._should_store_separately(content):
                    self._store_large_content(entry.id, content)
                    
            if metadata is not None:
                entry.meta_data.update(metadata)
                
            if vectors is not None:
                entry.vectors = vectors
                
            # Commit changes
            session.commit()
            
            # Update cache
            self._update_cache(entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()
            
    def delete(self, memory_id: int) -> bool:
        """
        Delete memory entry.
        
        Args:
            memory_id: Memory entry ID
            
        Returns:
            bool: Success status
        """
        try:
            # Create session
            session = self.Session()
            
            # Delete entry
            entry = session.query(MemoryEntry).filter_by(id=memory_id).first()
            
            if not entry:
                return False
                
            session.delete(entry)
            session.commit()
            
            # Remove from cache
            self.cache.pop(memory_id, None)
            
            # Remove separate storage if exists
            self._remove_large_content(memory_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()
            
    def _create_memory_dirs(self) -> None:
        """Create memory directories."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / 'large_objects').mkdir(exist_ok=True)
        
    def _should_store_separately(self, content: Dict[str, Any]) -> bool:
        """Check if content should be stored separately."""
        # Check size threshold (e.g., 1MB)
        return len(str(content)) > self.config.get('separate_storage_threshold', 1_000_000)
        
    def _store_large_content(self, memory_id: int, content: Dict[str, Any]) -> None:
        """Store large content separately."""
        filepath = self.memory_dir / 'large_objects' / f'{memory_id}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(content, f)
            
    def _load_large_content(self, memory_id: int) -> Dict[str, Any]:
        """Load separately stored content."""
        filepath = self.memory_dir / 'large_objects' / f'{memory_id}.pkl'
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return {}
        
    def _has_separate_storage(self, memory_id: int) -> bool:
        """Check if memory has separate storage."""
        filepath = self.memory_dir / 'large_objects' / f'{memory_id}.pkl'
        return filepath.exists()
        
    def _remove_large_content(self, memory_id: int) -> None:
        """Remove separately stored content."""
        filepath = self.memory_dir / 'large_objects' / f'{memory_id}.pkl'
        if filepath.exists():
            filepath.unlink()
            
    def _update_cache(self, entry: MemoryEntry) -> None:
        """Update cache with new entry."""
        self.cache[entry.id] = entry
        
        # Maintain cache size
        if len(self.cache) > self.cache_size:
            oldest = min(self.cache.items(), key=lambda x: x[1].timestamp)
            self.cache.pop(oldest[0])
            
    def _prepare_output(
        self,
        entry: MemoryEntry,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """Prepare memory entry for output."""
        output = {
            'id': entry.id,
            'type': entry.memory_type,
            'timestamp': entry.timestamp.isoformat(),
            'content': entry.content,
            'metadata': entry.meta_data
        }
        
        if include_vectors:
            output['vectors'] = entry.vectors
            
        return output
        
    def _vector_search(
        self,
        entries: List[MemoryEntry],
        query_vector: List[float],
        top_k: int
    ) -> List[MemoryEntry]:
        """Perform vector similarity search."""
        if not entries:
            return []
            
        # Convert query vector to numpy array
        query_vector = np.array(query_vector)
        
        # Calculate similarities
        similarities = []
        for entry in entries:
            if entry.vectors:
                # Use average of all vectors if multiple exist
                vectors = np.mean([v for v in entry.vectors.values()], axis=0)
                similarity = np.dot(query_vector, vectors) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vectors)
                )
                similarities.append((entry, similarity))
                
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:top_k]]