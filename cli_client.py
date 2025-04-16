#!/usr/bin/env python3
"""
KC-Riff CLI Client
A simple command-line client for interacting with the KC-Riff API server.
"""

import sys
import requests
import argparse
import json
from tabulate import tabulate
import time

# Constants
API_ENDPOINT = "http://localhost:5000"
MODELS_API = f"{API_ENDPOINT}/api/models"
DOWNLOAD_API = f"{API_ENDPOINT}/api/download"
REMOVE_API = f"{API_ENDPOINT}/api/remove"
STATUS_API = f"{API_ENDPOINT}/api/status"

def print_models(models, show_download_status=True):
    """Print a formatted table of models"""
    if not models:
        print("No models found.")
        return
    
    # Prepare table headers and rows
    headers = ["Name", "Description", "Size (MB)", "Parameters (B)", "Category", "Recommended"]
    if show_download_status:
        headers.append("Status")
    
    rows = []
    for model in models:
        row = [
            model.get("name", "Unknown"),
            model.get("description", "")[:50] + ("..." if len(model.get("description", "")) > 50 else ""),
            f"{model.get('size', 0) / 1000000:.1f}",
            f"{model.get('parameters', 0) / 1000000000:.1f}",
            model.get("category", "Unknown"),
            "✓" if model.get("kc_recommended", False) else ""
        ]
        
        if show_download_status:
            row.append("Downloaded" if model.get("is_downloaded", False) else "Not Downloaded")
        
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def list_models(args):
    """List all available models"""
    try:
        response = requests.get(MODELS_API)
        if response.status_code == 200:
            data = response.json()
            
            # Handle different categories of models
            all_models = data.get("models", [])
            recommended_models = [m for m in all_models if m.get("kc_recommended", False)]
            standard_models = [m for m in all_models if m.get("category", "") == "standard" and not m.get("kc_recommended", False)]
            advanced_models = [m for m in all_models if m.get("category", "") == "advanced" and not m.get("kc_recommended", False)]
            
            if args.recommended:
                print("\n== KillChaos Recommended Models ==")
                print_models(recommended_models)
            elif args.category:
                filtered_models = [m for m in all_models if m.get("category", "").lower() == args.category.lower()]
                print(f"\n== {args.category.title()} Models ==")
                print_models(filtered_models)
            else:
                print("\n== KillChaos Recommended Models ==")
                print_models(recommended_models)
                print("\n== Standard Models ==")
                print_models(standard_models)
                print("\n== Advanced Models ==")
                print_models(advanced_models)
            
            return True
        else:
            print(f"Error: API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error accessing API: {e}")
        return False

def download_model(args):
    """Download a model"""
    model_name = args.model
    if not model_name:
        print("Error: No model name provided.")
        return False
    
    try:
        # Start the download
        print(f"Starting download of model: {model_name}")
        response = requests.post(f"{DOWNLOAD_API}/{model_name}")
        
        if response.status_code != 200:
            print(f"Error: Failed to start download. API returned status code {response.status_code}")
            return False
        
        # Poll for status until complete
        print("Download initiated. Monitoring progress:")
        completed = False
        while not completed:
            try:
                status_response = requests.get(f"{STATUS_API}/{model_name}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if "status" in status_data:
                        if status_data["status"] == "downloading":
                            progress = status_data.get("progress", 0)
                            print(f"Progress: {progress:.1f}%", end="\r")
                        elif status_data["status"] == "completed":
                            completed = True
                            print("\nDownload completed successfully!")
                            return True
                        elif status_data["status"] == "failed":
                            print("\nDownload failed.")
                            return False
            except Exception as e:
                print(f"\nError checking status: {e}")
                return False
            
            # Sleep before next poll
            time.sleep(1)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def remove_model(args):
    """Remove a downloaded model"""
    model_name = args.model
    if not model_name:
        print("Error: No model name provided.")
        return False
    
    try:
        print(f"Removing model: {model_name}")
        response = requests.post(f"{REMOVE_API}/{model_name}")
        
        if response.status_code == 200:
            print("Model removed successfully.")
            return True
        else:
            print(f"Error: Failed to remove model. API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_ENDPOINT}/healthz")
        if response.status_code == 200:
            data = response.json()
            print(f"Server is running: {data.get('service', 'Unknown')} v{data.get('version', 'Unknown')}")
            return True
        else:
            print(f"Server is not responding correctly (status code: {response.status_code})")
            return False
    except Exception as e:
        print(f"Server not available: {e}")
        print(f"Make sure the server is running at {API_ENDPOINT}")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="KC-Riff Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--recommended", action="store_true", help="Show only recommended models")
    list_parser.add_argument("--category", help="Filter models by category (e.g., standard, advanced)")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Name of the model to download")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a downloaded model")
    remove_parser.add_argument("model", help="Name of the model to remove")
    
    # Check command
    subparsers.add_parser("check", help="Check if the server is running")
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the requested command
    if args.command == "list":
        if not list_models(args):
            return 1
    elif args.command == "download":
        if not download_model(args):
            return 1
    elif args.command == "remove":
        if not remove_model(args):
            return 1
    elif args.command == "check":
        if not check_server():
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())