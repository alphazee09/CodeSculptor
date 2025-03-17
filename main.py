"""
Main Application for AI Tool

This file serves as the entry point for the AI Tool application,
integrating all components including API connections, social media integration,
news analysis, frequency generator, and user interface.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import threading
import time

# Import all components
from api_connections import APIConnectionManager
from social_media_integration import SocialMediaManager, AccountManager
from news_analysis import NewsFetcher, TrendAnalyzer, EventPredictor, AdvancedAlgorithm
from frequency_generator import FrequencyGenerator
from user_interface import UserInterface

class AITool:
    """Main AI Tool application"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.initialized = False
        
        # Initialize components
        self.api_manager = None
        self.social_manager = None
        self.account_manager = None
        self.news_fetcher = None
        self.trend_analyzer = None
        self.event_predictor = None
        self.algorithm = None
        self.frequency_generator = None
        self.ui = None
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "ai_api_keys": {
                "manus": "YOUR_MANUS_API_KEY",
                "openai": "YOUR_OPENAI_API_KEY",
                "deepseek": "YOUR_DEEPSEEK_API_KEY"
            },
            "social_api_keys": {
                "twitter": "YOUR_TWITTER_API_KEY",
                "facebook": "YOUR_FACEBOOK_API_KEY",
                "instagram": "YOUR_INSTAGRAM_API_KEY",
                "tiktok": "YOUR_TIKTOK_API_KEY",
                "youtube": "YOUR_YOUTUBE_API_KEY",
                "linkedin": "YOUR_LINKEDIN_API_KEY"
            },
            "news_api_keys": {
                "newsapi": "YOUR_NEWSAPI_KEY",
                "gnews": "YOUR_GNEWS_API_KEY"
            },
            "app_settings": {
                "debug_mode": False,
                "log_level": "INFO",
                "max_threads": 10,
                "request_timeout": 30,  # seconds
                "cache_duration": 3600,  # seconds
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                    # Update default config with loaded values
                    if "ai_api_keys" in loaded_config:
                        default_config["ai_api_keys"].update(loaded_config["ai_api_keys"])
                    
                    if "social_api_keys" in loaded_config:
                        default_config["social_api_keys"].update(loaded_config["social_api_keys"])
                    
                    if "news_api_keys" in loaded_config:
                        default_config["news_api_keys"].update(loaded_config["news_api_keys"])
                    
                    if "app_settings" in loaded_config:
                        default_config["app_settings"].update(loaded_config["app_settings"])
                    
                print(f"Configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
        else:
            print("Using default configuration")
            
            # Save default config if no config file exists
            if config_path:
                try:
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, 'w') as f:
                        json.dump(default_config, f, indent=4)
                    print(f"Default configuration saved to {config_path}")
                except Exception as e:
                    print(f"Error saving default configuration: {e}")
        
        return default_config
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            print("Initializing AI Tool...")
            
            # Initialize API connections
            print("Initializing API connections...")
            self.api_manager = APIConnectionManager()
            
            # Create HTTP connectors for AI models
            for model, api_key in self.config["ai_api_keys"].items():
                if model == "manus":
                    self.api_manager.create_http_connector(model, api_key, "https://api.manus.ai/v1")
                elif model == "openai":
                    self.api_manager.create_http_connector(model, api_key, "https://api.openai.com/v1")
                elif model == "deepseek":
                    self.api_manager.create_http_connector(model, api_key, "https://api.deepseek.ai/v1")
            
            # Initialize social media integration
            print("Initializing social media integration...")
            self.social_manager = SocialMediaManager(self.api_manager)
            self.account_manager = AccountManager(self.social_manager)
            
            # Initialize news analysis
            print("Initializing news analysis...")
            self.news_fetcher = NewsFetcher(self.config["news_api_keys"].get("newsapi"))
            self.trend_analyzer = TrendAnalyzer(self.news_fetcher, self.social_manager)
            self.event_predictor = EventPredictor(self.trend_analyzer)
            self.algorithm = AdvancedAlgorithm(self.event_predictor)
            
            # Initialize frequency generator
            print("Initializing frequency generator...")
            self.frequency_generator = FrequencyGenerator()
            
            # Initialize user interface
            print("Initializing user interface...")
            self.ui = UserInterface()
            
            self.initialized = True
            print("AI Tool initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing AI Tool: {e}")
            return False
    
    def run(self) -> None:
        """Run the application"""
        if not self.initialized:
            if not self.initialize():
                print("Failed to initialize AI Tool")
                return
        
        try:
            print("Running AI Tool...")
            
            # Run the user interface
            self.ui.run()
            
            print("AI Tool is running")
            
            # Keep the application running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("AI Tool stopped")
        except Exception as e:
            print(f"Error running AI Tool: {e}")


def main():
    """Main entry point"""
    # Get config path from command line arguments
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and run the AI Tool
    ai_tool = AITool(config_path)
    ai_tool.run()


if __name__ == "__main__":
    main()
