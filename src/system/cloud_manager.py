"""
Cloud infrastructure manager for flexible hosting across different cloud providers.
"""
import logging
import boto3
import google.cloud.compute_v1 as google_compute
import azure.mgmt.compute as azure_compute
from azure.identity import DefaultAzureCredential
from typing import Dict, List, Any, Optional
import yaml
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class CloudManager:
    """
    Manages cloud infrastructure across multiple providers (AWS, GCP, Azure).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the cloud manager.
        
        Args:
            config_path: Path to cloud configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.clients = {}
        self.resources = {}
        self.initialized_providers = set()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cloud configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return {}
    
    def initialize_provider(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """
        Initialize a cloud provider client.
        
        Args:
            provider: Cloud provider name (aws, gcp, azure)
            credentials: Provider credentials
            
        Returns:
            Success status
        """
        try:
            if provider == 'aws':
                self.clients['aws'] = boto3.client(
                    'ec2',
                    aws_access_key_id=credentials.get('access_key'),
                    aws_secret_access_key=credentials.get('secret_key'),
                    region_name=credentials.get('region', 'us-east-1')
                )
                
            elif provider == 'gcp':
                self.clients['gcp'] = google_compute.InstancesClient()
                
            elif provider == 'azure':
                credential = DefaultAzureCredential()
                self.clients['azure'] = azure_compute.ComputeManagementClient(
                    credential=credential,
                    subscription_id=credentials.get('subscription_id')
                )
                
            else:
                logger.error(f"Unsupported provider: {provider}")
                return False
                
            self.initialized_providers.add(provider)
            logger.info(f"Initialized {provider} client")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {provider} client: {str(e)}")
            return False
    
    def create_instance(self, provider: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Create a cloud instance.
        
        Args:
            provider: Cloud provider name
            config: Instance configuration
            
        Returns:
            Instance ID if successful, None otherwise
        """
        try:
            if provider not in self.initialized_providers:
                raise ValueError(f"Provider {provider} not initialized")
                
            instance_id = None
            
            if provider == 'aws':
                response = self.clients['aws'].run_instances(
                    ImageId=config['image_id'],
                    InstanceType=config['instance_type'],
                    MinCount=1,
                    MaxCount=1,
                    SecurityGroupIds=config.get('security_groups', []),
                    SubnetId=config.get('subnet_id'),
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': k, 'Value': v} for k, v in config.get('tags', {}).items()]
                    }]
                )
                instance_id = response['Instances'][0]['InstanceId']
                
            elif provider == 'gcp':
                instance = google_compute.Instance()
                instance.name = config['name']
                instance.machine_type = config['machine_type']
                # Add more GCP-specific configuration
                
                operation = self.clients['gcp'].insert(
                    project=config['project_id'],
                    zone=config['zone'],
                    instance_resource=instance
                )
                operation.result()  # Wait for completion
                instance_id = instance.name
                
            elif provider == 'azure':
                # Azure instance creation logic
                pass
            
            if instance_id:
                self.resources[instance_id] = {
                    'provider': provider,
                    'type': 'instance',
                    'config': config
                }
                logger.info(f"Created {provider} instance {instance_id}")
                
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating {provider} instance: {str(e)}")
            return None
    
    def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate a cloud instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Success status
        """
        try:
            if instance_id not in self.resources:
                raise ValueError(f"Instance {instance_id} not found")
                
            resource = self.resources[instance_id]
            provider = resource['provider']
            
            if provider == 'aws':
                self.clients['aws'].terminate_instances(
                    InstanceIds=[instance_id]
                )
                
            elif provider == 'gcp':
                self.clients['gcp'].delete(
                    project=resource['config']['project_id'],
                    zone=resource['config']['zone'],
                    instance=instance_id
                ).result()
                
            elif provider == 'azure':
                # Azure instance termination logic
                pass
            
            del self.resources[instance_id]
            logger.info(f"Terminated {provider} instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error terminating instance {instance_id}: {str(e)}")
            return False
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get instance status.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Dictionary with instance status information
        """
        try:
            if instance_id not in self.resources:
                raise ValueError(f"Instance {instance_id} not found")
                
            resource = self.resources[instance_id]
            provider = resource['provider']
            
            if provider == 'aws':
                response = self.clients['aws'].describe_instances(
                    InstanceIds=[instance_id]
                )
                instance = response['Reservations'][0]['Instances'][0]
                return {
                    'state': instance['State']['Name'],
                    'public_ip': instance.get('PublicIpAddress'),
                    'private_ip': instance.get('PrivateIpAddress'),
                    'launch_time': instance['LaunchTime'].isoformat()
                }
                
            elif provider == 'gcp':
                instance = self.clients['gcp'].get(
                    project=resource['config']['project_id'],
                    zone=resource['config']['zone'],
                    instance=instance_id
                )
                return {
                    'state': instance.status,
                    'public_ip': instance.network_interfaces[0].access_configs[0].nat_ip,
                    'private_ip': instance.network_interfaces[0].network_ip,
                    'creation_time': instance.creation_timestamp
                }
                
            elif provider == 'azure':
                # Azure instance status logic
                pass
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting status for instance {instance_id}: {str(e)}")
            return {}
    
    def deploy_infrastructure(self, deployment_config: Dict[str, Any]) -> bool:
        """
        Deploy infrastructure across multiple providers.
        
        Args:
            deployment_config: Infrastructure deployment configuration
            
        Returns:
            Success status
        """
        try:
            # Initialize required providers
            for provider_config in deployment_config.get('providers', []):
                provider = provider_config['name']
                if provider not in self.initialized_providers:
                    if not self.initialize_provider(provider, provider_config['credentials']):
                        return False
            
            # Create instances
            for instance_config in deployment_config.get('instances', []):
                provider = instance_config.pop('provider')
                if not self.create_instance(provider, instance_config):
                    return False
            
            logger.info("Infrastructure deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying infrastructure: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up all cloud resources."""
        try:
            # Terminate all instances
            for instance_id in list(self.resources.keys()):
                if self.resources[instance_id]['type'] == 'instance':
                    self.terminate_instance(instance_id)
            
            # Clear clients
            self.clients.clear()
            self.initialized_providers.clear()
            logger.info("Cloud resources cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def export_infrastructure_state(self, path: str):
        """
        Export current infrastructure state to file.
        
        Args:
            path: Output file path
        """
        try:
            state = {
                'providers': list(self.initialized_providers),
                'resources': {}
            }
            
            # Export resource states
            for resource_id, resource in self.resources.items():
                if resource['type'] == 'instance':
                    state['resources'][resource_id] = {
                        'type': 'instance',
                        'provider': resource['provider'],
                        'status': self.get_instance_status(resource_id)
                    }
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Exported infrastructure state to {path}")
            
        except Exception as e:
            logger.error(f"Error exporting infrastructure state: {str(e)}")