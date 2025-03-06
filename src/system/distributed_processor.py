"""
Distributed processing system for parallel strategy testing and execution.
"""
import logging
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable
import asyncio
import distributed
from distributed import Client, LocalCluster
import concurrent.futures
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class DistributedProcessor:
    """
    Manages distributed processing for parallel strategy testing and execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the distributed processor.
        
        Args:
            config: Configuration settings for distributed processing
        """
        self.config = config or {}
        self.cluster = None
        self.client = None
        self.worker_processes = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.monitor_thread = None
        
    async def start(self) -> bool:
        """Start the distributed processing system."""
        try:
            n_workers = self.config.get('n_workers', mp.cpu_count())
            memory_limit = self.config.get('memory_limit', '4GB')
            
            # Create local cluster
            self.cluster = await LocalCluster(
                n_workers=n_workers,
                memory_limit=memory_limit,
                asynchronous=True
            ).__aenter__()
            
            # Create client
            self.client = await Client(self.cluster, asynchronous=True).__aenter__()
            
            # Start monitoring thread
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_system)
            self.monitor_thread.start()
            
            logger.info(f"Started distributed system with {n_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start distributed system: {str(e)}")
            return False
    
    async def stop(self):
        """Stop the distributed processing system."""
        try:
            self.is_running = False
            
            if self.monitor_thread:
                self.monitor_thread.join()
                
            if self.client:
                await self.client.close()
                self.client = None
                
            if self.cluster:
                await self.cluster.close()
                self.cluster = None
                
            # Clean up worker processes
            for process in self.worker_processes.values():
                process.terminate()
                process.join()
            
            self.worker_processes.clear()
            logger.info("Stopped distributed system")
            
        except Exception as e:
            logger.error(f"Error stopping distributed system: {str(e)}")
    
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for distributed processing.
        
        Args:
            task_func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        if not self.client:
            raise RuntimeError("Distributed system not started")
            
        try:
            future = await self.client.submit(task_func, *args, **kwargs)
            task_id = future.key
            
            self.task_queue.put({
                'id': task_id,
                'future': future,
                'submitted_at': time.time()
            })
            
            logger.info(f"Submitted task {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            raise
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a task."""
        try:
            # Find task in queue
            while True:
                try:
                    task = self.task_queue.get(timeout=timeout)
                    if task['id'] == task_id:
                        future = task['future']
                        break
                    else:
                        self.task_queue.put(task)
                except queue.Empty:
                    raise TimeoutError(f"Task {task_id} not found")
            
            result = await future
            
            self.result_queue.put({
                'id': task_id,
                'result': result,
                'completed_at': time.time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting result for task {task_id}: {str(e)}")
            raise
    
    def _monitor_system(self):
        """Monitor system health and task progress."""
        while self.is_running:
            try:
                # Check worker health
                worker_info = self.client.scheduler_info()['workers']
                for worker_id, info in worker_info.items():
                    if info['memory_limit'] - info['memory'] < 1e8:  # Less than 100MB free
                        logger.warning(f"Worker {worker_id} low on memory")
                
                # Check task progress
                all_tasks = self.client.processing()
                for worker_tasks in all_tasks.values():
                    for task_id, task_info in worker_tasks.items():
                        duration = time.time() - task_info['start_time']
                        if duration > 3600:  # More than 1 hour
                            logger.warning(f"Task {task_id} running for {duration/3600:.1f} hours")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(60)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.client:
            return {}
            
        try:
            metrics = {
                'n_workers': len(self.client.scheduler_info()['workers']),
                'tasks_pending': len(self.task_queue.queue),
                'tasks_completed': self.result_queue.qsize(),
                'memory_usage': {},
                'cpu_usage': {}
            }
            
            worker_info = self.client.scheduler_info()['workers']
            for worker_id, info in worker_info.items():
                metrics['memory_usage'][worker_id] = info['memory'] / info['memory_limit']
                metrics['cpu_usage'][worker_id] = info['cpu']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
