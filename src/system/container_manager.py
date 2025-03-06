"""
Container management system for portable deployment.
"""
import logging
import docker
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import time
import os

logger = logging.getLogger(__name__)

class ContainerManager:
    """
    Manages Docker containers for portable deployment of the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the container manager.
        
        Args:
            config_path: Path to container configuration file
        """
        self.client = docker.from_env()
        self.config = self._load_config(config_path) if config_path else {}
        self.containers = {}
        self.networks = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load container configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return {}
    
    def build_image(self, name: str, dockerfile_path: str, context_path: str) -> bool:
        """
        Build a Docker image.
        
        Args:
            name: Image name and tag
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Building image {name}")
            self.client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=name
            )
            return True
            
        except Exception as e:
            logger.error(f"Error building image {name}: {str(e)}")
            return False
    
    def create_network(self, name: str) -> Optional[str]:
        """
        Create a Docker network.
        
        Args:
            name: Network name
            
        Returns:
            Network ID if successful, None otherwise
        """
        try:
            network = self.client.networks.create(
                name,
                driver="bridge",
                check_duplicate=True
            )
            self.networks[name] = network
            return network.id
            
        except Exception as e:
            logger.error(f"Error creating network {name}: {str(e)}")
            return None
    
    def start_container(self, name: str, image: str, **kwargs) -> Optional[str]:
        """
        Start a Docker container.
        
        Args:
            name: Container name
            image: Image name
            **kwargs: Additional container configuration
            
        Returns:
            Container ID if successful, None otherwise
        """
        try:
            # Add default configuration
            container_config = {
                'name': name,
                'detach': True,
                'restart_policy': {"Name": "unless-stopped"},
                'healthcheck': {
                    "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                    "interval": 30000000000,  # 30 seconds
                    "timeout": 5000000000,    # 5 seconds
                    "retries": 3
                }
            }
            
            # Update with provided configuration
            container_config.update(kwargs)
            
            # Create and start container
            container = self.client.containers.run(
                image=image,
                **container_config
            )
            
            self.containers[name] = container
            logger.info(f"Started container {name} ({container.id})")
            
            return container.id
            
        except Exception as e:
            logger.error(f"Error starting container {name}: {str(e)}")
            return None
    
    def stop_container(self, name: str) -> bool:
        """
        Stop a Docker container.
        
        Args:
            name: Container name
            
        Returns:
            Success status
        """
        try:
            if name in self.containers:
                container = self.containers[name]
                container.stop()
                container.remove()
                del self.containers[name]
                logger.info(f"Stopped and removed container {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error stopping container {name}: {str(e)}")
            return False
    
    def get_container_logs(self, name: str, lines: int = 100) -> Optional[str]:
        """
        Get container logs.
        
        Args:
            name: Container name
            lines: Number of log lines to retrieve
            
        Returns:
            Log content if successful, None otherwise
        """
        try:
            if name in self.containers:
                container = self.containers[name]
                return container.logs(tail=lines).decode('utf-8')
            return None
            
        except Exception as e:
            logger.error(f"Error getting logs for container {name}: {str(e)}")
            return None
    
    def get_container_status(self, name: str) -> Dict[str, Any]:
        """
        Get container status.
        
        Args:
            name: Container name
            
        Returns:
            Dictionary with container status information
        """
        try:
            if name in self.containers:
                container = self.containers[name]
                container.reload()  # Refresh container info
                
                return {
                    'id': container.id,
                    'status': container.status,
                    'health': container.attrs['State']['Health']['Status'] 
                            if 'Health' in container.attrs['State'] else 'N/A',
                    'created': container.attrs['Created'],
                    'ports': container.ports,
                    'network': container.attrs['NetworkSettings']['Networks']
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting status for container {name}: {str(e)}")
            return {}
    
    def deploy_system(self, deployment_config: Dict[str, Any]) -> bool:
        """
        Deploy the complete system using configuration.
        
        Args:
            deployment_config: System deployment configuration
            
        Returns:
            Success status
        """
        try:
            # Create network if specified
            if 'network' in deployment_config:
                network_name = deployment_config['network']
                if not self.create_network(network_name):
                    return False
            
            # Deploy containers
            for container_config in deployment_config.get('containers', []):
                name = container_config.pop('name')
                image = container_config.pop('image')
                
                # Build image if build context is provided
                if 'build' in container_config:
                    build_config = container_config.pop('build')
                    if not self.build_image(
                        image,
                        build_config['dockerfile'],
                        build_config['context']
                    ):
                        return False
                
                # Start container
                if not self.start_container(name, image, **container_config):
                    return False
            
            logger.info("System deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying system: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up all containers and networks."""
        try:
            # Stop and remove all containers
            for name in list(self.containers.keys()):
                self.stop_container(name)
            
            # Remove all networks
            for name, network in self.networks.items():
                try:
                    network.remove()
                except Exception as e:
                    logger.error(f"Error removing network {name}: {str(e)}")
                    
            self.networks.clear()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def export_container_config(self, path: str):
        """
        Export current container configuration to file.
        
        Args:
            path: Output file path
        """
        try:
            config = {
                'containers': [],
                'networks': list(self.networks.keys())
            }
            
            # Export container configurations
            for name, container in self.containers.items():
                container_config = {
                    'name': name,
                    'image': container.image.tags[0],
                    'ports': container.ports,
                    'environment': container.attrs['Config']['Env'],
                    'volumes': container.attrs['HostConfig']['Binds'],
                    'networks': list(container.attrs['NetworkSettings']['Networks'].keys())
                }
                config['containers'].append(container_config)
            
            # Save to file
            with open(path, 'w') as f:
                yaml.safe_dump(config, f)
                
            logger.info(f"Exported container configuration to {path}")
            
        except Exception as e:
            logger.error(f"Error exporting container configuration: {str(e)}")