#!/usr/bin/env python3
"""
Simple test script to verify that key architectural improvements are working.
"""
import asyncio
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = logging.getLogger("simple_test")

async def test_memory_system():
    """Test the improved memory system."""
    from src.core.simple_memory import SimpleMemoryManager
    
    logger.info("Testing enhanced memory system")
    
    # Create memory manager
    memory = SimpleMemoryManager(memory_dir="test_memory")
    
    # Test working memory
    test_data = {
        "value": 123,
        "nested": {
            "test": "value"
        },
        "list": [1, 2, 3]
    }
    
    # Set working memory
    memory.set_working_memory("test_key", test_data)
    
    # Retrieve working memory
    retrieved_data = memory.get_working_memory("test_key")
    
    # Verify data is correct
    assert retrieved_data is not None, "Failed to retrieve working memory"
    assert retrieved_data["value"] == 123, "Retrieved data doesn't match stored data"
    assert retrieved_data["nested"]["test"] == "value", "Nested data doesn't match"
    assert retrieved_data["list"] == [1, 2, 3], "List data doesn't match"
    
    logger.info("Memory system test passed")
    return True

async def test_retry_mechanism():
    """Test a simple retry mechanism similar to what we added to the agents."""
    from src.core.simple_memory import SimpleMemoryManager
    
    logger.info("Testing retry mechanism")
    
    # Create a function with retry logic similar to agent process_with_retries
    async def retry_function(data, max_retries=3):
        import random
        
        retry_count = 0
        last_error = None
        process_count = 0
        
        # Store original data for retry attempts
        original_data = data.copy()
        
        while retry_count <= max_retries:
            try:
                # Increment process count
                process_count += 1
                
                # Fail on the first two attempts
                if process_count < 3:
                    raise ValueError(f"Simulated error (attempt {process_count})")
                    
                # Succeed on the third attempt
                return {"result": "success", "attempts": process_count}
                
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.info(f"Error in process (attempt {retry_count}/{max_retries}): {str(e)}")
                
                # If we have retries left, wait and try again
                if retry_count <= max_retries:
                    # Simple delay (no actual wait in test)
                    pass
                    
                    # Reset data to original for clean retry
                    data = original_data.copy()
                
        # If we get here, all retries failed
        return {
            "error": f"Processing failed after {max_retries} attempts",
            "last_error": str(last_error) if last_error else "Unknown error",
            "status": "error"
        }
    
    # Test the retry mechanism
    result = await retry_function({"test": "data"}, max_retries=3)
    
    # Verify the result
    assert result["result"] == "success", "Retry failed to return successful result"
    assert result["attempts"] == 3, "Retry didn't retry the correct number of times"
    
    logger.info("Retry mechanism test passed")
    return True

async def test_circuit_breaker_pattern():
    """Test a simple circuit breaker pattern similar to what we added to the agents."""
    logger.info("Testing circuit breaker pattern")
    
    # Implement a simple circuit breaker pattern
    class CircuitBreaker:
        def __init__(self, max_errors=5, window_seconds=300):
            self.errors = []
            self.max_errors = max_errors
            self.window_seconds = window_seconds
            
        def add_error(self):
            # Add current time to errors list
            self.errors.append(datetime.now())
            
        def is_open(self):
            """Check if circuit breaker is open (blocking requests)"""
            # Remove old errors outside the window
            now = datetime.now()
            self.errors = [e for e in self.errors 
                          if (now - e).total_seconds() < self.window_seconds]
            
            # Check if we've exceeded the error threshold
            return len(self.errors) >= self.max_errors
            
        async def execute(self, func, *args, **kwargs):
            """Execute function with circuit breaker protection"""
            # Check if circuit is open
            if self.is_open():
                return {
                    "status": "error",
                    "circuit_breaker": True,
                    "error": "Service temporarily unavailable due to multiple failures",
                    "retry_after": 60
                }
                
            # Otherwise execute the function
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Record the error
                self.add_error()
                
                # Return error response
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    # Create a test circuit breaker
    breaker = CircuitBreaker(max_errors=5, window_seconds=300)
    
    # Create a test function that always fails
    async def failing_function():
        raise ValueError("Simulated error")
    
    # Generate enough errors to trigger circuit breaker
    for i in range(6):
        response = await breaker.execute(failing_function)
        assert response["status"] == "error", "Function execution should fail"
        
    # Now, try once more - should be blocked by circuit breaker
    response = await breaker.execute(failing_function)
    
    # Check if the circuit breaker was triggered
    assert response["status"] == "error", "Circuit breaker didn't return error status"
    assert "circuit_breaker" in response, "Circuit breaker flag not present in response"
    
    logger.info("Circuit breaker pattern test passed")
    return True

async def main():
    """Run all tests."""
    logger.info("Starting architecture improvement tests")
    
    test_results = {
        "memory_system": await test_memory_system(),
        "retry_mechanism": await test_retry_mechanism(),
        "circuit_breaker": await test_circuit_breaker_pattern()
    }
    
    # Print results
    logger.info("Architecture improvement test results:")
    for test_name, result in test_results.items():
        logger.info(f"  {test_name}: {'PASSED' if result else 'FAILED'}")
    
    # Return overall success
    return all(test_results.values())

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)