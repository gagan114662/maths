#!/usr/bin/env python
"""
QuantConnect API Integration Module

This module provides functionality to interact with the QuantConnect API for:
1. Uploading and compiling algorithms
2. Running backtests and retrieving results
3. Managing QuantConnect projects and files
4. Analyzing backtest performance

It uses the QuantConnect.API Python package to interact with the platform.
"""

import os
import sys
import json
import time
import logging
import hashlib
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    # Import the QuantConnect API
    from quantconnect.api import Api
except ImportError:
    logger.error("QuantConnect API package not found. Please install it with 'pip install quantconnect'")
    # Define a placeholder Api class for development without the actual package
    class Api:
        def __init__(self, user_id=None, token=None):
            self.user_id = user_id
            self.token = token
            logger.warning("Using placeholder QuantConnect API class. Functionality will be limited.")
        
        def create_project(self, name, description=None):
            return {"success": True, "projects": [{"projectId": "placeholder_project_id"}]}
        
        def list_projects(self):
            return {"success": True, "projects": []}
        
        def read_project_files(self, project_id):
            return {"success": True, "files": {}}
        
        def add_project_file(self, project_id, name, content):
            return {"success": True}
        
        def update_project_file_content(self, project_id, name, content):
            return {"success": True}
        
        def delete_project_file(self, project_id, name):
            return {"success": True}
        
        def create_compile(self, project_id):
            return {"success": True, "compileId": "placeholder_compile_id"}
        
        def read_compile(self, project_id, compile_id):
            return {"success": True, "state": "success"}
        
        def create_backtest(self, project_id, compile_id, name=None):
            return {"success": True, "backtest": {"backtestId": "placeholder_backtest_id"}}
        
        def read_backtest(self, project_id, backtest_id):
            return {"success": True, "state": "completed", "statistics": {}}
        
        def list_backtests(self, project_id):
            return {"success": True, "backtests": []}
        
        def delete_backtest(self, project_id, backtest_id):
            return {"success": True}

class QuantConnectAPIClient:
    """Client for interacting with the QuantConnect API."""
    
    def __init__(self, user_id=None, token=None, config_path=None):
        """
        Initialize the QuantConnect API client.
        
        Args:
            user_id (str): QuantConnect user ID
            token (str): QuantConnect API token
            config_path (str): Path to configuration file with user_id and token
        """
        # Load user_id and token from config file if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    user_id = config.get('user_id', user_id)
                    token = config.get('token', token)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        # Get from environment variables if not provided
        if not user_id:
            user_id = os.environ.get('QC_USER_ID')
        if not token:
            token = os.environ.get('QC_API_TOKEN')
        
        if not user_id or not token:
            logger.warning("QuantConnect user ID or token not provided. Some functionality may be limited.")
        
        # Initialize API client
        self.api = Api(user_id=user_id, token=token)
        self.user_id = user_id
        self.default_project_name = "Mathematricks Integration"
    
    def _handle_response(self, response, operation):
        """Handle API response and log appropriate messages."""
        if response['success']:
            logger.info(f"Successfully completed {operation}")
            return response
        else:
            error_msg = response.get('errors', ['Unknown error'])[0]
            logger.error(f"Failed to {operation}: {error_msg}")
            raise Exception(f"QuantConnect API error: {error_msg}")
    
    def create_project(self, name=None, description=None):
        """
        Create a new project on QuantConnect.
        
        Args:
            name (str): Project name
            description (str): Project description
            
        Returns:
            str: Project ID
        """
        name = name or f"{self.default_project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = description or "Project generated by Mathematricks Integration"
        
        response = self.api.create_project(name, description)
        response = self._handle_response(response, f"create project '{name}'")
        
        project_id = response['projects'][0]['projectId']
        logger.info(f"Created project {name} with ID {project_id}")
        return project_id
    
    def get_or_create_project(self, name=None):
        """
        Get an existing project by name or create a new one.
        
        Args:
            name (str): Project name
            
        Returns:
            str: Project ID
        """
        name = name or self.default_project_name
        
        # List existing projects
        projects_response = self.api.list_projects()
        projects_response = self._handle_response(projects_response, "list projects")
        
        # Find project by name
        for project in projects_response['projects']:
            if project['name'] == name:
                logger.info(f"Found existing project {name} with ID {project['projectId']}")
                return project['projectId']
        
        # Create new project if not found
        return self.create_project(name)
    
    def upload_algorithm(self, project_id, file_path):
        """
        Upload an algorithm file to a QuantConnect project.
        
        Args:
            project_id (str): Project ID
            file_path (str): Path to algorithm file
            
        Returns:
            bool: True if successful
        """
        # Read algorithm file
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading algorithm file {file_path}: {e}")
            raise
        
        # Get file name
        file_name = os.path.basename(file_path)
        
        # Check if file already exists in project
        files_response = self.api.read_project_files(project_id)
        files_response = self._handle_response(files_response, f"read files in project {project_id}")
        
        if file_name in files_response['files']:
            # Update existing file
            response = self.api.update_project_file_content(project_id, file_name, content)
            response = self._handle_response(response, f"update file {file_name} in project {project_id}")
        else:
            # Add new file
            response = self.api.add_project_file(project_id, file_name, content)
            response = self._handle_response(response, f"add file {file_name} to project {project_id}")
        
        return True
    
    def compile_algorithm(self, project_id, timeout=60):
        """
        Compile an algorithm in a project.
        
        Args:
            project_id (str): Project ID
            timeout (int): Maximum time to wait for compilation (seconds)
            
        Returns:
            str: Compile ID if successful
        """
        # Start compilation
        compile_response = self.api.create_compile(project_id)
        compile_response = self._handle_response(compile_response, f"start compilation for project {project_id}")
        
        compile_id = compile_response['compileId']
        
        # Wait for compilation to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = self.api.read_compile(project_id, compile_id)
            status_response = self._handle_response(status_response, f"check compilation status for {compile_id}")
            
            if status_response['state'] == 'success':
                logger.info(f"Compilation successful for project {project_id}")
                return compile_id
            
            if status_response['state'] == 'failed':
                error_msg = status_response.get('logs', 'Unknown compilation error')
                logger.error(f"Compilation failed for project {project_id}: {error_msg}")
                raise Exception(f"Compilation failed: {error_msg}")
            
            # Wait before checking again
            time.sleep(2)
        
        logger.error(f"Compilation timed out after {timeout} seconds for project {project_id}")
        raise Exception(f"Compilation timed out after {timeout} seconds")
    
    def run_backtest(self, project_id, compile_id, name=None, timeout=300):
        """
        Run a backtest on a compiled algorithm.
        
        Args:
            project_id (str): Project ID
            compile_id (str): Compile ID
            name (str): Backtest name
            timeout (int): Maximum time to wait for backtest (seconds)
            
        Returns:
            dict: Backtest results
        """
        # Generate backtest name if not provided
        if not name:
            name = f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start backtest
        backtest_response = self.api.create_backtest(project_id, compile_id, name)
        backtest_response = self._handle_response(backtest_response, f"start backtest {name}")
        
        backtest_id = backtest_response['backtest']['backtestId']
        
        # Wait for backtest to complete
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = self.api.read_backtest(project_id, backtest_id)
            status_response = self._handle_response(status_response, f"check backtest status for {backtest_id}")
            
            if status_response['state'] == 'completed':
                logger.info(f"Backtest completed successfully for {backtest_id}")
                return status_response
            
            if status_response['state'] == 'failed':
                error_msg = status_response.get('error', 'Unknown backtest error')
                logger.error(f"Backtest failed for {backtest_id}: {error_msg}")
                raise Exception(f"Backtest failed: {error_msg}")
            
            # Wait before checking again
            time.sleep(5)
        
        logger.error(f"Backtest timed out after {timeout} seconds for {backtest_id}")
        raise Exception(f"Backtest timed out after {timeout} seconds")
    
    def extract_backtest_metrics(self, backtest_results):
        """
        Extract key performance metrics from backtest results.
        
        Args:
            backtest_results (dict): Raw backtest results from API
            
        Returns:
            dict: Structured performance metrics
        """
        statistics = backtest_results.get('statistics', {})
        
        # Extract key metrics
        metrics = {
            'total_trades': int(statistics.get('Total Trades', 0)),
            'win_rate': float(statistics.get('Win Rate', 0)),
            'annual_return': float(statistics.get('Annual Return', 0)),
            'sharpe_ratio': float(statistics.get('Sharpe Ratio', 0)),
            'equity_final': float(statistics.get('Final Equity', 0)),
            'max_drawdown': float(statistics.get('Max Drawdown', 0)),
            'alpha': float(statistics.get('Alpha', 0)),
            'beta': float(statistics.get('Beta', 0)),
            'information_ratio': float(statistics.get('Information Ratio', 0)),
            'volatility': float(statistics.get('Volatility', 0)),
            'avg_win': float(statistics.get('Average Win', 0)),
            'avg_loss': float(statistics.get('Average Loss', 0))
        }
        
        return metrics
    
    def save_results(self, metrics, output_path=None):
        """
        Save backtest metrics to a file.
        
        Args:
            metrics (dict): Performance metrics
            output_path (str): Path to save results
            
        Returns:
            str: Path to saved results
        """
        # Generate output path if not provided
        if not output_path:
            output_dir = os.path.join(os.getcwd(), 'qc_results')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Save metrics to file
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved backtest results to {output_path}")
        return output_path
    
    def run_algorithm(self, algorithm_path, project_name=None, backtest_name=None):
        """
        Run a full workflow: upload, compile, and backtest an algorithm.
        
        Args:
            algorithm_path (str): Path to algorithm file
            project_name (str): Project name
            backtest_name (str): Backtest name
            
        Returns:
            dict: Structured performance metrics
        """
        # Get or create project
        project_id = self.get_or_create_project(project_name)
        
        # Upload algorithm
        self.upload_algorithm(project_id, algorithm_path)
        
        # Compile algorithm
        compile_id = self.compile_algorithm(project_id)
        
        # Run backtest
        backtest_results = self.run_backtest(project_id, compile_id, backtest_name)
        
        # Extract and save metrics
        metrics = self.extract_backtest_metrics(backtest_results)
        results_path = self.save_results(metrics)
        
        metrics['file_path'] = results_path
        
        return metrics

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantConnect API integration")
    parser.add_argument("--user-id", help="QuantConnect user ID")
    parser.add_argument("--token", help="QuantConnect API token")
    parser.add_argument("--config", help="Path to config file with user_id and token")
    parser.add_argument("--algorithm", required=True, help="Path to algorithm file")
    parser.add_argument("--project", help="Project name")
    parser.add_argument("--backtest", help="Backtest name")
    parser.add_argument("--output", help="Output path for results")
    
    args = parser.parse_args()
    
    # Initialize API client
    client = QuantConnectAPIClient(args.user_id, args.token, args.config)
    
    # Run algorithm
    metrics = client.run_algorithm(args.algorithm, args.project, args.backtest)
    
    # Save results to specified output path if provided
    if args.output:
        client.save_results(metrics, args.output)
    
    # Print summary
    print("\nQuantConnect Backtest Results:")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")

if __name__ == "__main__":
    main()