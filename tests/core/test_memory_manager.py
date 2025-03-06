"""
Tests for Memory Manager.
"""
import pytest
import json
import pickle
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.memory_manager import MemoryManager, MemoryEntry, Base

@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def memory_manager(test_db, tmp_path):
    """Create Memory Manager instance with test configuration."""
    config = {
        'db_path': 'sqlite:///:memory:',
        'memory_dir': str(tmp_path),
        'cache_size': 10,
        'separate_storage_threshold': 1000
    }
    
    with patch('src.core.memory_manager.load_config', return_value=config):
        manager = MemoryManager()
        manager.engine = test_db
        manager.Session = sessionmaker(bind=test_db)
        return manager

def test_store_basic(memory_manager):
    """Test basic memory storage."""
    content = {'test': 'data'}
    memory_id = memory_manager.store('test_type', content)
    
    assert memory_id is not None
    stored = memory_manager.retrieve(memory_id)
    assert stored['content'] == content

def test_store_with_metadata(memory_manager):
    """Test storage with metadata."""
    content = {'test': 'data'}
    metadata = {'source': 'test', 'priority': 'high'}
    memory_id = memory_manager.store('test_type', content, metadata=metadata)
    
    stored = memory_manager.retrieve(memory_id)
    assert stored['metadata'] == metadata

def test_store_with_vectors(memory_manager):
    """Test storage with vector embeddings."""
    content = {'test': 'data'}
    vectors = {'embedding': [0.1, 0.2, 0.3]}
    memory_id = memory_manager.store('test_type', content, vectors=vectors)
    
    stored = memory_manager.retrieve(memory_id, include_vectors=True)
    assert np.allclose(stored['vectors']['embedding'], vectors['embedding'])

def test_large_content_storage(memory_manager):
    """Test storage of large content."""
    large_content = {'data': 'x' * 2000}  # Exceeds threshold
    memory_id = memory_manager.store('test_type', large_content)
    
    # Verify separate storage
    large_file = Path(memory_manager.memory_dir) / 'large_objects' / f'{memory_id}.pkl'
    assert large_file.exists()
    
    # Verify retrieval
    stored = memory_manager.retrieve(memory_id)
    assert stored['content'] == large_content

def test_search_by_type(memory_manager):
    """Test memory search by type."""
    # Store test entries
    memory_manager.store('type1', {'data': 1})
    memory_manager.store('type1', {'data': 2})
    memory_manager.store('type2', {'data': 3})
    
    results = memory_manager.search(memory_type='type1')
    assert len(results) == 2
    assert all(r['type'] == 'type1' for r in results)

def test_search_by_metadata(memory_manager):
    """Test memory search by metadata."""
    memory_manager.store('test', {'data': 1}, metadata={'tag': 'A'})
    memory_manager.store('test', {'data': 2}, metadata={'tag': 'B'})
    
    results = memory_manager.search(metadata_filter={'tag': 'A'})
    assert len(results) == 1
    assert results[0]['metadata']['tag'] == 'A'

def test_vector_search(memory_manager):
    """Test vector similarity search."""
    # Store entries with vectors
    vectors1 = {'v': [1.0, 0.0, 0.0]}
    vectors2 = {'v': [0.0, 1.0, 0.0]}
    vectors3 = {'v': [0.0, 0.0, 1.0]}
    
    memory_manager.store('test', {'data': 1}, vectors=vectors1)
    memory_manager.store('test', {'data': 2}, vectors=vectors2)
    memory_manager.store('test', {'data': 3}, vectors=vectors3)
    
    # Search with query vector
    query = [1.0, 0.1, 0.1]
    results = memory_manager.search(vector_query=query, top_k=1)
    
    assert len(results) == 1
    assert results[0]['content']['data'] == 1  # Should match vectors1

def test_update_memory(memory_manager):
    """Test memory update."""
    # Store initial entry
    memory_id = memory_manager.store('test', {'initial': 'data'})
    
    # Update content
    new_content = {'updated': 'data'}
    success = memory_manager.update(memory_id, content=new_content)
    
    assert success
    updated = memory_manager.retrieve(memory_id)
    assert updated['content'] == new_content

def test_delete_memory(memory_manager):
    """Test memory deletion."""
    # Store entry
    memory_id = memory_manager.store('test', {'test': 'data'})
    
    # Delete entry
    success = memory_manager.delete(memory_id)
    
    assert success
    assert memory_manager.retrieve(memory_id) is None

def test_cache_management(memory_manager):
    """Test cache management."""
    # Fill cache
    for i in range(15):  # Exceeds cache size
        memory_manager.store('test', {'data': i})
        
    assert len(memory_manager.cache) <= memory_manager.cache_size

def test_separate_storage_management(memory_manager):
    """Test separate storage management."""
    large_content = {'data': 'x' * 2000}
    memory_id = memory_manager.store('test', large_content)
    
    # Verify file exists
    file_path = Path(memory_manager.memory_dir) / 'large_objects' / f'{memory_id}.pkl'
    assert file_path.exists()
    
    # Delete memory
    memory_manager.delete(memory_id)
    
    # Verify file is removed
    assert not file_path.exists()

def test_memory_persistence(memory_manager, tmp_path):
    """Test memory persistence across sessions."""
    # Store in first session
    memory_id = memory_manager.store('test', {'data': 'persistent'})
    
    # Create new session
    new_manager = MemoryManager()
    new_manager.engine = memory_manager.engine
    new_manager.Session = memory_manager.Session
    
    # Verify data persists
    stored = new_manager.retrieve(memory_id)
    assert stored['content']['data'] == 'persistent'

def test_error_handling(memory_manager):
    """Test error handling."""
    # Test invalid memory ID
    assert memory_manager.retrieve(999) is None
    
    # Test invalid update
    assert not memory_manager.update(999, content={'test': 'data'})
    
    # Test invalid delete
    assert not memory_manager.delete(999)

def test_vector_operations(memory_manager):
    """Test vector-related operations."""
    vectors = {
        'embedding1': [0.1, 0.2, 0.3],
        'embedding2': [0.4, 0.5, 0.6]
    }
    
    # Store with multiple vectors
    memory_id = memory_manager.store('test', {'data': 'vector_test'}, vectors=vectors)
    
    # Retrieve with vectors
    stored = memory_manager.retrieve(memory_id, include_vectors=True)
    assert 'vectors' in stored
    assert len(stored['vectors']) == 2
    assert np.allclose(stored['vectors']['embedding1'], vectors['embedding1'])

def test_bulk_operations(memory_manager):
    """Test bulk operations performance."""
    # Bulk store
    entries = [
        ('test', {'data': i}, {'meta': i}, None)
        for i in range(100)
    ]
    
    for entry in entries:
        memory_manager.store(*entry)
    
    # Bulk search
    results = memory_manager.search(memory_type='test', top_k=50)
    assert len(results) == 50

def test_concurrent_access(memory_manager):
    """Test concurrent access patterns."""
    from concurrent.futures import ThreadPoolExecutor
    
    def store_and_retrieve(i):
        memory_id = memory_manager.store('test', {'data': i})
        return memory_manager.retrieve(memory_id)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(store_and_retrieve, range(10)))
    
    assert len(results) == 10
    assert all(r is not None for r in results)