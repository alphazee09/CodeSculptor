"""
Social Media Integration Module for AI Tool

This file implements the social media integration components for the AI tool,
providing interfaces for connecting to various social media platforms and
retrieving data from them using the API connections framework.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import sys

# Import API connections
from api_connections import APIConnectionManager, TwitterAPIConnector, LinkedInAPIConnector

class SocialMediaPlatform:
    """Base class for social media platform integrations"""
    
    def __init__(self, name: str, api_connector: Any):
        self.name = name
        self.api_connector = api_connector
        
    def get_platform_name(self) -> str:
        """Get the name of the platform"""
        return self.name
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors from the social media platform"""
        return {
            "success": False,
            "error": str(error),
            "platform": self.name,
            "timestamp": datetime.now().isoformat()
        }


class TwitterPlatform(SocialMediaPlatform):
    """Twitter platform integration"""
    
    def __init__(self, twitter_connector: TwitterAPIConnector):
        super().__init__("twitter", twitter_connector)
        
    def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get Twitter user profile information"""
        try:
            result = self.api_connector.get_user_profile(username)
            
            if not result.get("success", False):
                return result
            
            # Process and format the data
            profile_data = result.get("data", {})
            user_data = self._extract_user_data(profile_data)
            
            return {
                "success": True,
                "platform": self.name,
                "username": username,
                "profile_data": user_data,
                "raw_data": profile_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _extract_user_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user data from the Twitter profile response"""
        try:
            # Navigate through the nested structure to get user data
            user_result = profile_data.get("result", {}).get("data", {}).get("user", {}).get("result", {})
            legacy_data = user_result.get("legacy", {})
            
            return {
                "name": legacy_data.get("name", ""),
                "screen_name": legacy_data.get("screen_name", ""),
                "description": legacy_data.get("description", ""),
                "followers_count": legacy_data.get("followers_count", 0),
                "friends_count": legacy_data.get("friends_count", 0),
                "statuses_count": legacy_data.get("statuses_count", 0),
                "location": legacy_data.get("location", ""),
                "verified": legacy_data.get("verified", False),
                "is_blue_verified": user_result.get("is_blue_verified", False),
                "profile_image_url": legacy_data.get("profile_image_url_https", ""),
                "created_at": legacy_data.get("created_at", "")
            }
        except Exception as e:
            print(f"Error extracting user data: {e}")
            return {}
    
    def get_user_tweets(self, username: str, count: int = 20) -> Dict[str, Any]:
        """Get tweets from a user"""
        try:
            # First get the user profile to get the user ID
            profile_result = self.get_user_profile(username)
            
            if not profile_result.get("success", False):
                return profile_result
            
            # Extract the user ID from the raw data
            user_id = self._extract_user_id(profile_result.get("raw_data", {}))
            
            if not user_id:
                return {
                    "success": False,
                    "error": f"Could not extract user ID for {username}",
                    "platform": self.name,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get the user's tweets
            tweets_result = self.api_connector.get_user_tweets(user_id, count)
            
            if not tweets_result.get("success", False):
                return tweets_result
            
            # Process and format the tweets
            tweets_data = tweets_result.get("data", {})
            tweets = self._extract_tweets(tweets_data)
            
            return {
                "success": True,
                "platform": self.name,
                "username": username,
                "user_id": user_id,
                "tweets": tweets,
                "raw_data": tweets_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _extract_user_id(self, profile_data: Dict[str, Any]) -> str:
        """Extract user ID from the Twitter profile response"""
        try:
            # Navigate through the nested structure to get user ID
            return profile_data.get("result", {}).get("data", {}).get("user", {}).get("result", {}).get("rest_id", "")
        except Exception as e:
            print(f"Error extracting user ID: {e}")
            return ""
    
    def _extract_tweets(self, tweets_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tweets from the Twitter response"""
        try:
            # This would involve navigating through the complex Twitter API response
            # For simplicity, we'll return a placeholder
            # In a real implementation, this would parse the actual tweet data
            
            # Get the timeline instructions
            timeline = tweets_data.get("result", {}).get("timeline", {})
            instructions = timeline.get("instructions", [])
            
            tweets = []
            for instruction in instructions:
                if instruction.get("type") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        # Process each entry to extract tweet data
                        tweet_data = self._process_tweet_entry(entry)
                        if tweet_data:
                            tweets.append(tweet_data)
            
            return tweets
        except Exception as e:
            print(f"Error extracting tweets: {e}")
            return []
    
    def _process_tweet_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a tweet entry from the Twitter response"""
        try:
            # This is a simplified implementation
            # In a real implementation, this would extract all relevant tweet data
            content = entry.get("content", {})
            item_content = content.get("itemContent", {})
            tweet_results = item_content.get("tweet_results", {})
            result = tweet_results.get("result", {})
            
            if not result:
                return None
            
            legacy = result.get("legacy", {})
            
            return {
                "id": legacy.get("id_str", ""),
                "text": legacy.get("full_text", ""),
                "created_at": legacy.get("created_at", ""),
                "retweet_count": legacy.get("retweet_count", 0),
                "favorite_count": legacy.get("favorite_count", 0),
                "reply_count": legacy.get("reply_count", 0),
                "quote_count": legacy.get("quote_count", 0)
            }
        except Exception as e:
            print(f"Error processing tweet entry: {e}")
            return None
    
    def search_tweets(self, query: str, count: int = 20, search_type: str = "Top") -> Dict[str, Any]:
        """Search for tweets"""
        try:
            result = self.api_connector.search_twitter(query, count, search_type)
            
            if not result.get("success", False):
                return result
            
            # Process and format the search results
            search_data = result.get("data", {})
            tweets = self._extract_search_results(search_data)
            
            return {
                "success": True,
                "platform": self.name,
                "query": query,
                "search_type": search_type,
                "tweets": tweets,
                "raw_data": search_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _extract_search_results(self, search_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search results from the Twitter response"""
        try:
            # Similar to _extract_tweets, but for search results
            # For simplicity, we'll use the same approach
            timeline = search_data.get("result", {}).get("timeline", {})
            instructions = timeline.get("instructions", [])
            
            tweets = []
            for instruction in instructions:
                if "entries" in instruction:
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        # Process each entry to extract tweet data
                        tweet_data = self._process_tweet_entry(entry)
                        if tweet_data:
                            tweets.append(tweet_data)
            
            return tweets
        except Exception as e:
            print(f"Error extracting search results: {e}")
            return []


class LinkedInPlatform(SocialMediaPlatform):
    """LinkedIn platform integration"""
    
    def __init__(self, linkedin_connector: LinkedInAPIConnector):
        super().__init__("linkedin", linkedin_connector)
        
    def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get LinkedIn user profile information"""
        try:
            result = self.api_connector.get_user_profile(username)
            
            if not result.get("success", False):
                return result
            
            # Process and format the data
            profile_data = result.get("data", {})
            user_data = self._extract_user_data(profile_data)
            
            return {
                "success": True,
                "platform": self.name,
                "username": username,
                "profile_data": user_data,
                "raw_data": profile_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _extract_user_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user data from the LinkedIn profile response"""
        try:
            # Navigate through the nested structure to get user data
            data = profile_data.get("data", {})
            
            # Extract profile information
            profile_info = {}
            
            # Extract post information if available
            post = data.get("post", {})
            if post:
                author = post.get("author", {})
                profile_info = {
                    "id": author.get("id", ""),
                    "firstName": author.get("firstName", ""),
                    "lastName": author.get("lastName", ""),
                    "headline": author.get("headline", ""),
                    "url": author.get("url", "")
                }
            
            return profile_info
        except Exception as e:
            print(f"Error extracting user data: {e}")
            return {}
    
    def search_people(self, keywords: str, count: int = 10) -> Dict[str, Any]:
        """Search for people on LinkedIn"""
        try:
            result = self.api_connector.search_people(keywords)
            
            if not result.get("success", False):
                return result
            
            # Process and format the search results
            search_data = result.get("data", {})
            people = self._extract_search_results(search_data)
            
            # Limit to requested count
            people = people[:count] if len(people) > count else people
            
            return {
                "success": True,
                "platform": self.name,
                "keywords": keywords,
                "people": people,
                "raw_data": search_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return self.handle_error(e)
    
    def _extract_search_results(self, search_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search results from the LinkedIn response"""
        try:
            # Navigate through the nested structure to get search results
            data = search_data.get("data", {})
            items = data.get("items", [])
            
            people = []
            for item in items:
                people.append({
                    "fullName": item.get("fullName", ""),
                    "headline": item.get("headline", ""),
                    "location": item.get("location", ""),
                    "profileURL": item.get("profileURL", ""),
                    "username": item.get("username", "")
                })
            
            return people
        except Exception as e:
            print(f"Error extracting search results: {e}")
            return []


class SocialMediaManager:
    """Manages social media platform integrations"""
    
    def __init__(self, api_manager: APIConnectionManager):
        self.api_manager = api_manager
        self.platforms = {}
        
        # Initialize built-in platforms
        self._initialize_platforms()
        
    def _initialize_platforms(self):
        """Initialize built-in social media platforms"""
        # Twitter
        twitter = TwitterPlatform(self.api_manager.twitter)
        self.platforms["twitter"] = twitter
        
        # LinkedIn
        linkedin = LinkedInPlatform(self.api_manager.linkedin)
        self.platforms["linkedin"] = linkedin
        
        # Additional platforms would be initialized here
        
    def get_platform(self, name: str) -> Optional[SocialMediaPlatform]:
        """Get a social media platform by name"""
        return self.platforms.get(name.lower())
    
    def register_platform(self, name: str, platform: SocialMediaPlatform) -> None:
        """Register a social media platform"""
        self.platforms[name.lower()] = platform
    
    def get_all_platforms(self) -> Dict[str, SocialMediaPlatform]:
        """Get all registered social media platforms"""
        return self.platforms


class UserAccount:
    """Represents a user account on a social media platform"""
    
    def __init__(self, platform: str, username: str, access_token: str = None):
        self.platform = platform
        self.username = username
        self.access_token = access_token
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.profile_data = {}
        
    def update_profile_data(self, profile_data: Dict[str, Any]) -> None:
        """Update the profile data"""
        self.profile_data = profile_data
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "platform": self.platform,
            "username": self.username,
            "access_token": self.access_token,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "profile_data": self.profile_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserAccount':
        """Create from dictionary"""
        account = cls(
            platform=data.get("platform", ""),
            username=data.get("username", ""),
            access_token=data.get("access_token")
        )
        
        # Set additional properties
        account.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        account.last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        account.profile_data = data.get("profile_data", {})
        
        return account


class AccountManager:
    """Manages user accounts for social media platforms"""
    
    def __init__(self, social_media_manager: SocialMediaManager):
        self.social_media_manager = social_media_manager
        self.accounts = {}  # platform -> username -> account
        
    def add_account(self, platform: str, username: str, access_token: str = None) -> bool:
        """Add a user account"""
        try:
            # Create the account
            account = UserAccount(platform, username, access_token)
            
            # Add to accounts
            if platform not in self.accounts:
                self.accounts[platform] = {}
            
            self.accounts[platform][username] = account
            
            # Try to fetch profile data
            self._update_account_profile(account)
            
            return True
        except Exception as e:
            print(f"Error adding account: {e}")
            return False
    
    def _update_account_profile(self, account: UserAccount) -> bool:
        """Update account profile data"""
        try:
            platform_obj = self.social_media_manager.get_platform(account.platform)
            
            if not platform_obj:
                print(f"Platform not found: {account.platform}")
                return False
            
            # Get profile data
            result = platform_obj.get_user_profile(account.username)
            
            if not result.get("success", False):
                print(f"Error getting profile data: {result.get('error', 'Unknown error')}")
                return False
            
            # Update account profile data
            account.update_profile_data(result.get("profile_data", {}))
            
            return True
        except Exception as e:
            print(f"Error updating account profile: {e}")
            return False
    
    def remove_account(self, platform: str, username: str) -> bool:
        """Remove a user account"""
        try:
            if platform in self.accounts and username in self.accounts[platform]:
                del self.accounts[platform][username]
                
                # Remove platform entry if empty
                if not self.accounts[platform]:
                    del self.accounts[platform]
                
                return True
            
            return False
        except Exception as e:
            print(f"Error removing account: {e}")
            return False
    
    def get_account(self, platform: str, username: str) -> Optional[UserAccount]:
        """Get a user account"""
        try:
            if platform in self.accounts and username in self.accounts[platform]:
                return self.accounts[platform][username]
            
            return None
        except Exception as e:
            print(f"Error getting account: {e}")
            return None
    
    def get_all_accounts(self) -> Dict[str, Dict[str, UserAccount]]:
        """Get all user accounts"""
        return self.accounts
    
    def get_platform_accounts(self, platform: str) -> Dict[str, UserAccount]:
        """Get all accounts for a platform"""
        return self.accounts.get(platform, {})
    
    def save_accounts(self, filepath: str) -> bool:
        """Save accounts to a file"""
        try:
            # Convert accounts to serializable format
            serialized = {}
            for platform, platform_accounts in self.accounts.items():
                serialized[platform] = {}
                for username, account in platform_accounts.items():
                    serialized[platform][username] = account.to_dict()
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serialized, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving accounts: {e}")
            return False
    
    def load_accounts(self, filepath: str) -> bool:
        """Load accounts from a file"""
        try:
            # Load from file
            with open(filepath, 'r') as f:
                serialized = json.load(f)
            
            # Convert to accounts
            for platform, platform_accounts in serialized.items():
                if platform not in self.accounts:
                    self.accounts[platform] = {}
                
                for username, account_data in platform_accounts.items():
                    self.accounts[platform][username] = UserAccount.from_dict(account_data)
            
            return True
        except Exception as e:
            print(f"Error loading accounts: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create API connection manager
    api_manager = APIConnectionManager()
    
    # Create social media manager
    social_manager = SocialMediaManager(api_manager)
    
    # Create account manager
    account_manager = AccountManager(social_manager)
    
    # Add accounts
    account_manager.add_account("twitter", "elonmusk")
    account_manager.add_account("linkedin", "adamselipsky")
    
    # Get Twitter platform
    twitter = social_manager.get_platform("twitter")
    if twitter:
        # Search tweets
        search_result = twitter.search_tweets("AI news", count=5)
        print(f"Twitter search success: {search_result.get('success', False)}")
        
        # Get user profile
        profile_result = twitter.get_user_profile("elonmusk")
        print(f"Twitter profile success: {profile_result.get('success', False)}")
    
    # Get LinkedIn platform
    linkedin = social_manager.get_platform("linkedin")
    if linkedin:
        # Search people
        search_result = linkedin.search_people("CEO", count=5)
        print(f"LinkedIn search success: {search_result.get('success', False)}")
        
        # Get user profile
        profile_result = linkedin.get_user_profile("adamselipsky")
        print(f"LinkedIn profile success: {profile_result.get('success', False)}")
