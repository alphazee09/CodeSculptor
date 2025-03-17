"""
API Connections Framework for AI Tool

This file implements the API connection framework for the AI tool,
providing standardized interfaces for connecting to various AI models
and external services.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
import sys

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')

class APIConnector(ABC):
    """Base abstract class for all API connectors"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    @abstractmethod
    def make_request(self, endpoint: str, method: str = "GET", 
                    params: Dict[str, Any] = None, 
                    data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the API"""
        pass
    
    def handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle the API response"""
        try:
            response.raise_for_status()
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.content else {},
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": str(e),
                "response_text": response.text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle general errors"""
        return {
            "success": False,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }


class HTTPConnector(APIConnector):
    """HTTP-based API connector implementation"""
    
    def make_request(self, endpoint: str, method: str = "GET", 
                    params: Dict[str, Any] = None, 
                    data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an HTTP request to the API"""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, params=params, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=self.headers, params=params, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, params=params)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}",
                    "timestamp": datetime.now().isoformat()
                }
            
            return self.handle_response(response)
        except Exception as e:
            return self.handle_error(e)


class DataSourceConnector:
    """Connector for Manus Data API sources"""
    
    def __init__(self):
        try:
            from data_api import ApiClient
            self.client = ApiClient()
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: data_api module not available")
    
    def call_api(self, api_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call a data API endpoint"""
        if not self.available:
            return {
                "success": False,
                "error": "Data API module not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = self.client.call_api(api_name, query=query)
            return {
                "success": True,
                "api_name": api_name,
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "api_name": api_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class TwitterAPIConnector:
    """Specialized connector for Twitter API using DataSourceConnector"""
    
    def __init__(self, data_source_connector: DataSourceConnector):
        self.connector = data_source_connector
        self.platform = "twitter"
    
    def search_twitter(self, query: str, count: int = 20, 
                      search_type: str = "Top", cursor: str = None) -> Dict[str, Any]:
        """Search Twitter for tweets matching a query"""
        params = {
            "query": query,
            "count": count,
            "type": search_type
        }
        
        if cursor:
            params["cursor"] = cursor
            
        return self.connector.call_api("Twitter/search_twitter", params)
    
    def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get Twitter user profile by username"""
        params = {
            "username": username
        }
        
        return self.connector.call_api("Twitter/get_user_profile_by_username", params)
    
    def get_user_tweets(self, user_id: str, count: int = 20, cursor: str = None) -> Dict[str, Any]:
        """Get tweets from a specific user by ID"""
        params = {
            "user": user_id,
            "count": count
        }
        
        if cursor:
            params["cursor"] = cursor
            
        return self.connector.call_api("Twitter/get_user_tweets", params)


class LinkedInAPIConnector:
    """Specialized connector for LinkedIn API using DataSourceConnector"""
    
    def __init__(self, data_source_connector: DataSourceConnector):
        self.connector = data_source_connector
        self.platform = "linkedin"
    
    def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get LinkedIn user profile by username"""
        params = {
            "username": username
        }
        
        return self.connector.call_api("LinkedIn/get_user_profile_by_username", params)
    
    def search_people(self, keywords: str, start: str = None, 
                     first_name: str = None, last_name: str = None,
                     school: str = None, title: str = None, 
                     company: str = None) -> Dict[str, Any]:
        """Search for people on LinkedIn"""
        params = {
            "keywords": keywords
        }
        
        # Add optional parameters if provided
        if start:
            params["start"] = start
        if first_name:
            params["firstName"] = first_name
        if last_name:
            params["lastName"] = last_name
        if school:
            params["keywordSchool"] = school
        if title:
            params["keywordTitle"] = title
        if company:
            params["company"] = company
            
        return self.connector.call_api("LinkedIn/search_people", params)


class APIConnectionManager:
    """Manages API connections for the AI tool"""
    
    def __init__(self):
        self.connectors = {}
        self.data_source = DataSourceConnector()
        
        # Initialize specialized connectors
        self.twitter = TwitterAPIConnector(self.data_source)
        self.linkedin = LinkedInAPIConnector(self.data_source)
        
    def register_connector(self, name: str, connector: APIConnector) -> None:
        """Register an API connector"""
        self.connectors[name] = connector
        
    def get_connector(self, name: str) -> Optional[APIConnector]:
        """Get an API connector by name"""
        return self.connectors.get(name)
    
    def create_http_connector(self, name: str, api_key: str, base_url: str) -> APIConnector:
        """Create and register an HTTP-based API connector"""
        connector = HTTPConnector(api_key, base_url)
        self.register_connector(name, connector)
        return connector


# Example usage
if __name__ == "__main__":
    # Create API connection manager
    api_manager = APIConnectionManager()
    
    # Create HTTP connectors for AI models
    openai_connector = api_manager.create_http_connector(
        "openai", 
        "YOUR_OPENAI_API_KEY", 
        "https://api.openai.com/v1"
    )
    
    # Use Twitter connector
    twitter_result = api_manager.twitter.search_twitter("AI news", count=5)
    print(f"Twitter search success: {twitter_result.get('success', False)}")
    
    # Use LinkedIn connector
    linkedin_result = api_manager.linkedin.get_user_profile("adamselipsky")
    print(f"LinkedIn profile success: {linkedin_result.get('success', False)}")
