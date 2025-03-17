"""
News Analysis Module for AI Tool

This file implements the news analysis components for the AI tool,
providing capabilities to fetch news, analyze trends, and predict
potentially dangerous events using an advanced algorithm.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import sys
import random
import math
from collections import Counter

# Import API connections
from api_connections import APIConnectionManager
from social_media_integration import SocialMediaManager

class NewsFetcher:
    """Fetches news from various sources"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.sources = [
            "bbc-news", "cnn", "reuters", "associated-press", 
            "the-washington-post", "the-wall-street-journal",
            "bloomberg", "financial-times", "the-economist"
        ]
        self.categories = [
            "business", "entertainment", "general", "health", 
            "science", "sports", "technology"
        ]
        self.base_url = "https://newsapi.org/v2"
        
    def fetch_top_headlines(self, country: str = "us", category: str = None, 
                           sources: List[str] = None, query: str = None, 
                           page_size: int = 20, page: int = 1) -> Dict[str, Any]:
        """Fetch top headlines"""
        try:
            # In a real implementation, this would make an API call to a news service
            # For this blueprint, we'll simulate the response
            
            # Build parameters
            params = {
                "country": country,
                "pageSize": page_size,
                "page": page
            }
            
            if category:
                params["category"] = category
            
            if sources:
                params["sources"] = ",".join(sources)
            
            if query:
                params["q"] = query
            
            # Simulate API call
            articles = self._generate_simulated_articles(
                count=page_size,
                query=query,
                category=category,
                sources=sources
            )
            
            return {
                "success": True,
                "status": "ok",
                "totalResults": len(articles) * 5,  # Simulate more pages
                "articles": articles,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def fetch_everything(self, query: str, sources: List[str] = None,
                        domains: List[str] = None, from_date: str = None,
                        to_date: str = None, language: str = "en",
                        sort_by: str = "publishedAt", page_size: int = 20,
                        page: int = 1) -> Dict[str, Any]:
        """Fetch everything matching the query"""
        try:
            # In a real implementation, this would make an API call to a news service
            # For this blueprint, we'll simulate the response
            
            # Build parameters
            params = {
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
                "page": page
            }
            
            if sources:
                params["sources"] = ",".join(sources)
            
            if domains:
                params["domains"] = ",".join(domains)
            
            if from_date:
                params["from"] = from_date
            
            if to_date:
                params["to"] = to_date
            
            # Simulate API call
            articles = self._generate_simulated_articles(
                count=page_size,
                query=query,
                sources=sources
            )
            
            return {
                "success": True,
                "status": "ok",
                "totalResults": len(articles) * 10,  # Simulate more pages
                "articles": articles,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def fetch_sources(self, category: str = None, language: str = "en",
                     country: str = None) -> Dict[str, Any]:
        """Fetch available sources"""
        try:
            # In a real implementation, this would make an API call to a news service
            # For this blueprint, we'll return a simulated list of sources
            
            # Build parameters
            params = {
                "language": language
            }
            
            if category:
                params["category"] = category
            
            if country:
                params["country"] = country
            
            # Filter sources based on parameters
            filtered_sources = self.sources
            
            if category:
                # In a real implementation, this would filter based on actual source categories
                # For this blueprint, we'll just take a subset
                filtered_sources = filtered_sources[:5]
            
            # Generate source objects
            sources = []
            for source_id in filtered_sources:
                sources.append({
                    "id": source_id,
                    "name": source_id.replace("-", " ").title(),
                    "description": f"Description for {source_id}",
                    "url": f"https://{source_id.replace('-', '')}.com",
                    "category": category or random.choice(self.categories),
                    "language": language,
                    "country": country or "us"
                })
            
            return {
                "success": True,
                "status": "ok",
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_simulated_articles(self, count: int, query: str = None,
                                   category: str = None, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Generate simulated news articles"""
        articles = []
        
        # Use provided sources or default sources
        article_sources = sources if sources else self.sources
        
        # Use provided category or random categories
        categories = [category] if category else self.categories
        
        # Generate articles
        for i in range(count):
            # Select source
            source_id = article_sources[i % len(article_sources)]
            source_name = source_id.replace("-", " ").title()
            
            # Select category
            article_category = categories[i % len(categories)]
            
            # Generate title
            title_prefix = ""
            if query:
                title_prefix = f"{query}: "
            
            title = f"{title_prefix}News article {i+1} about {article_category}"
            
            # Generate content
            content = f"This is the content of article {i+1} about {article_category}."
            if query:
                content += f" It is related to {query}."
            
            # Generate published date (within the last week)
            days_ago = random.randint(0, 7)
            published_at = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            # Create article object
            article = {
                "source": {
                    "id": source_id,
                    "name": source_name
                },
                "author": f"Author {i+1}",
                "title": title,
                "description": f"Description of article {i+1} about {article_category}",
                "url": f"https://example.com/news/{i+1}",
                "urlToImage": f"https://example.com/images/{i+1}.jpg",
                "publishedAt": published_at,
                "content": content
            }
            
            articles.append(article)
        
        return articles


class TrendAnalyzer:
    """Analyzes trending topics from news and social media"""
    
    def __init__(self, news_fetcher: NewsFetcher, social_media_manager: SocialMediaManager):
        self.news_fetcher = news_fetcher
        self.social_media_manager = social_media_manager
        self.trend_cache = {}
        self.cache_expiry = 3600  # seconds
        
    def analyze_trends(self, timeframe: str = "day", count: int = 10) -> Dict[str, Any]:
        """Analyze trending topics from news and social media"""
        try:
            # Check cache
            cache_key = f"trends_{timeframe}_{count}"
            if cache_key in self.trend_cache:
                cache_entry = self.trend_cache[cache_key]
                cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
                
                if cache_age < self.cache_expiry:
                    return cache_entry["data"]
            
            # Fetch news
            news_result = self._fetch_news_for_timeframe(timeframe, count)
            
            # Fetch trends from social media
            social_trends = self._fetch_social_trends(count)
            
            # Extract topics from news
            news_topics = self._extract_topics_from_news(news_result)
            
            # Combine and analyze trends
            combined_trends = self._combine_trends(news_topics, social_trends, count)
            
            # Prepare result
            result = {
                "success": True,
                "timeframe": timeframe,
                "news_trends": news_topics,
                "social_trends": social_trends,
                "combined_trends": combined_trends,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            self.trend_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now()
            }
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _fetch_news_for_timeframe(self, timeframe: str, count: int) -> Dict[str, Any]:
        """Fetch news for the specified timeframe"""
        # Convert timeframe to date parameters
        to_date = datetime.now().isoformat()
        from_date = None
        
        if timeframe == "day":
            from_date = (datetime.now() - timedelta(days=1)).isoformat()
        elif timeframe == "week":
            from_date = (datetime.now() - timedelta(days=7)).isoformat()
        elif timeframe == "month":
            from_date = (datetime.now() - timedelta(days=30)).isoformat()
        
        # Fetch news
        return self.news_fetcher.fetch_everything(
            query="",  # Empty query to get all news
            from_date=from_date,
            to_date=to_date,
            page_size=count * 3  # Fetch more to ensure we have enough for analysis
        )
    
    def _fetch_social_trends(self, count: int) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch trends from social media platforms"""
        social_trends = {}
        
        # Get Twitter trends
        twitter = self.social_media_manager.get_platform("twitter")
        if twitter:
            try:
                # In a real implementation, this would call a Twitter API endpoint for trends
                # For this blueprint, we'll simulate it
                twitter_trends = []
                for i in range(count):
                    twitter_trends.append({
                        "name": f"#TwitterTrend{i+1}",
                        "query": f"TwitterTrend{i+1}",
                        "tweet_volume": random.randint(1000, 100000)
                    })
                
                social_trends["twitter"] = twitter_trends
            except Exception as e:
                social_trends["twitter"] = {"error": str(e)}
        
        # Get LinkedIn trends
        linkedin = self.social_media_manager.get_platform("linkedin")
        if linkedin:
            try:
                # In a real implementation, this would call a LinkedIn API endpoint for trends
                # For this blueprint, we'll simulate it
                linkedin_trends = []
                for i in range(count):
                    linkedin_trends.append({
                        "name": f"LinkedIn Trend {i+1}",
                        "query": f"LinkedInTrend{i+1}",
                        "engagement_count": random.randint(500, 50000)
                    })
                
                social_trends["linkedin"] = linkedin_trends
            except Exception as e:
                social_trends["linkedin"] = {"error": str(e)}
        
        return social_trends
    
    def _extract_topics_from_news(self, news_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract topics from news articles"""
        if not news_result.get("success", False):
            return []
        
        articles = news_result.get("articles", [])
        
        # Extract keywords from titles and content
        keywords = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            
            # In a real implementation, this would use NLP to extract keywords
            # For this blueprint, we'll use a simple approach
            words = title.lower().split() + content.lower().split()
            words = [word for word in words if len(word) > 3]  # Filter short words
            keywords.extend(words)
        
        # Count keyword frequencies
        keyword_counts = Counter(keywords)
        
        # Convert to topics
        topics = []
        for keyword, count in keyword_counts.most_common(20):
            topics.append({
                "topic": keyword,
                "count": count,
                "source": "news"
            })
        
        return topics
    
    def _combine_trends(self, news_topics: List[Dict[str, Any]], 
                       social_trends: Dict[str, List[Dict[str, Any]]],
                       count: int) -> List[Dict[str, Any]]:
        """Combine trends from news and social media"""
        # Create a dictionary to track combined trends
        combined = {}
        
        # Add news topics
        for topic in news_topics:
            topic_name = topic["topic"]
            if topic_name not in combined:
                combined[topic_name] = {
                    "topic": topic_name,
                    "news_volume": topic["count"],
                    "social_volume": 0,
                    "sources": ["news"]
                }
        
        # Add social trends
        for platform, trends in social_trends.items():
            if isinstance(trends, list):
                for trend in trends:
                    topic_name = trend.get("name", "").lower()
                    
                    if topic_name in combined:
                        # Update existing topic
                        combined[topic_name]["social_volume"] += trend.get("tweet_volume", 0) + trend.get("engagement_count", 0)
                        if platform not in combined[topic_name]["sources"]:
                            combined[topic_name]["sources"].append(platform)
                    else:
                        # Add new topic
                        combined[topic_name] = {
                            "topic": topic_name,
                            "news_volume": 0,
                            "social_volume": trend.get("tweet_volume", 0) + trend.get("engagement_count", 0),
                            "sources": [platform]
                        }
        
        # Calculate total volume and sort
        for topic_name, topic in combined.items():
            topic["total_volume"] = topic["news_volume"] + topic["social_volume"]
            
            # Calculate sentiment (simulated)
            topic["sentiment"] = random.uniform(0.0, 1.0)
            
            # Calculate momentum (simulated)
            topic["momentum"] = random.uniform(0.0, 1.0)
        
        # Sort by total volume
        sorted_topics = sorted(
            combined.values(),
            key=lambda x: x["total_volume"],
            reverse=True
        )
        
        # Return top N
        return sorted_topics[:count]


class EventPredictor:
    """Predicts potential dangerous events based on news and trends"""
    
    def __init__(self, trend_analyzer: TrendAnalyzer):
        self.trend_analyzer = trend_analyzer
        self.risk_keywords = [
            "crisis", "danger", "threat", "risk", "warning", "alert",
            "emergency", "disaster", "catastrophe", "hazard", "conflict",
            "attack", "violence", "terrorism", "war", "outbreak", "epidemic",
            "pandemic", "earthquake", "hurricane", "tornado", "flood", "fire",
            "explosion", "collapse", "crash", "accident", "leak", "spill",
            "contamination", "pollution", "shortage", "outage", "failure"
        ]
        
    def predict_events(self, timeframe: str = "week", threshold: float = 0.7) -> Dict[str, Any]:
        """Predict potential dangerous events based on trends"""
        try:
            # Get trends
            trends_result = self.trend_analyzer.analyze_trends(timeframe=timeframe)
            
            if not trends_result["success"]:
                return trends_result
            
            # Analyze trends for potential dangerous events
            potential_events = []
            for trend in trends_result["combined_trends"]:
                # Calculate risk score
                risk_score = self._calculate_risk_score(trend)
                
                if risk_score > threshold:
                    # Create event prediction
                    event = {
                        "topic": trend["topic"],
                        "risk_score": risk_score,
                        "confidence": self._calculate_confidence(trend),
                        "potential_impact": self._determine_impact(trend),
                        "timeframe": self._determine_event_timeframe(trend),
                        "sources": trend.get("sources", []),
                        "news_volume": trend.get("news_volume", 0),
                        "social_volume": trend.get("social_volume", 0)
                    }
                    
                    potential_events.append(event)
            
            return {
                "success": True,
                "timeframe": timeframe,
                "threshold": threshold,
                "potential_events": potential_events,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_risk_score(self, trend: Dict[str, Any]) -> float:
        """Calculate risk score for a trend"""
        # Base score from sentiment and momentum
        base_score = (1.0 - trend.get("sentiment", 0.5)) * 0.5 + trend.get("momentum", 0.5) * 0.5
        
        # Adjust based on keyword matching
        keyword_factor = 0.0
        for keyword in self.risk_keywords:
            if keyword in trend["topic"].lower():
                keyword_factor = 0.3
                break
        
        # Adjust based on volume
        volume_factor = min(0.2, (trend.get("total_volume", 0) / 10000) * 0.2)
        
        # Combine factors
        risk_score = base_score + keyword_factor + volume_factor
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_confidence(self, trend: Dict[str, Any]) -> float:
        """Calculate confidence in the prediction"""
        # Base confidence
        base_confidence = 0.5
        
        # Adjust based on sources
        sources = trend.get("sources", [])
        source_factor = min(0.3, len(sources) * 0.1)
        
        # Adjust based on volumes
        news_volume = trend.get("news_volume", 0)
        social_volume = trend.get("social_volume", 0)
        
        volume_factor = min(0.2, (news_volume / 100) * 0.1 + (social_volume / 10000) * 0.1)
        
        # Combine factors
        confidence = base_confidence + source_factor + volume_factor
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, confidence))
    
    def _determine_impact(self, trend: Dict[str, Any]) -> List[str]:
        """Determine potential impact areas"""
        # In a real implementation, this would use more sophisticated analysis
        # For this blueprint, we'll use a simple approach
        
        impact_areas = []
        
        # Check for economic impact
        economic_keywords = ["economy", "market", "stock", "financial", "trade", "business"]
        for keyword in economic_keywords:
            if keyword in trend["topic"].lower():
                impact_areas.append("economic")
                break
        
        # Check for social impact
        social_keywords = ["social", "community", "public", "people", "population"]
        for keyword in social_keywords:
            if keyword in trend["topic"].lower():
                impact_areas.append("social")
                break
        
        # Check for political impact
        political_keywords = ["political", "government", "policy", "election", "vote"]
        for keyword in political_keywords:
            if keyword in trend["topic"].lower():
                impact_areas.append("political")
                break
        
        # Check for environmental impact
        environmental_keywords = ["environment", "climate", "pollution", "ecosystem"]
        for keyword in environmental_keywords:
            if keyword in trend["topic"].lower():
                impact_areas.append("environmental")
                break
        
        # Add default if none found
        if not impact_areas:
            impact_areas = ["general"]
        
        return impact_areas
    
    def _determine_event_timeframe(self, trend: Dict[str, Any]) -> str:
        """Determine the timeframe for the potential event"""
        # In a real implementation, this would use more sophisticated analysis
        # For this blueprint, we'll use momentum as a proxy
        
        momentum = trend.get("momentum", 0.5)
        
        if momentum > 0.8:
            return "immediate"
        elif momentum > 0.5:
            return "short-term"
        else:
            return "long-term"


class AdvancedAlgorithm:
    """Advanced algorithm for news analysis and event prediction"""
    
    def __init__(self, event_predictor: EventPredictor):
        self.event_predictor = event_predictor
        self.historical_data = []
        self.model_weights = {
            "trend_momentum": 0.3,
            "sentiment_analysis": 0.2,
            "historical_patterns": 0.3,
            "social_amplification": 0.2
        }
        
    def train_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the algorithm model with historical data"""
        try:
            # In a real implementation, this would involve machine learning
            # For this blueprint, we'll simulate it
            self.historical_data = training_data
            
            # Adjust weights based on training data
            if training_data:
                # Simulate weight adjustment
                self.model_weights["trend_momentum"] = 0.3 + random.uniform(-0.05, 0.05)
                self.model_weights["sentiment_analysis"] = 0.2 + random.uniform(-0.05, 0.05)
                self.model_weights["historical_patterns"] = 0.3 + random.uniform(-0.05, 0.05)
                self.model_weights["social_amplification"] = 0.2 + random.uniform(-0.05, 0.05)
                
                # Normalize weights
                total = sum(self.model_weights.values())
                for key in self.model_weights:
                    self.model_weights[key] /= total
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def analyze_and_predict(self, timeframe: str = "month", 
                           confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Analyze current trends and predict potential dangerous events"""
        try:
            # Get initial predictions
            initial_predictions = self.event_predictor.predict_events(
                timeframe=timeframe, 
                threshold=0.5  # Lower threshold to get more candidates
            )
            
            if not initial_predictions["success"]:
                return initial_predictions
            
            # Apply advanced algorithm to refine predictions
            refined_predictions = []
            for event in initial_predictions["potential_events"]:
                # Apply model weights to calculate refined risk score
                refined_score = (
                    event["risk_score"] * self.model_weights["trend_momentum"] +
                    event["confidence"] * self.model_weights["sentiment_analysis"] +
                    self._calculate_historical_factor(event) * self.model_weights["historical_patterns"] +
                    self._calculate_social_factor(event) * self.model_weights["social_amplification"]
                )
                
                if refined_score > confidence_threshold:
                    # Create refined prediction
                    refined_event = event.copy()
                    refined_event["risk_score"] = refined_score
                    refined_event["confidence"] = event["confidence"] * 1.2  # Boosted confidence
                    refined_event["analysis_factors"] = {
                        "trend_momentum": event["risk_score"] * self.model_weights["trend_momentum"],
                        "sentiment_analysis": event["confidence"] * self.model_weights["sentiment_analysis"],
                        "historical_patterns": self._calculate_historical_factor(event) * self.model_weights["historical_patterns"],
                        "social_amplification": self._calculate_social_factor(event) * self.model_weights["social_amplification"]
                    }
                    refined_event["recommended_actions"] = self._generate_recommendations(refined_event)
                    
                    refined_predictions.append(refined_event)
            
            return {
                "success": True,
                "timeframe": timeframe,
                "confidence_threshold": confidence_threshold,
                "predictions": refined_predictions,
                "model_weights": self.model_weights,
                "algorithm_version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_historical_factor(self, event: Dict[str, Any]) -> float:
        """Calculate historical factor based on historical data"""
        # In a real implementation, this would analyze historical patterns
        # For this blueprint, we'll simulate it
        
        # Check if we have historical data
        if not self.historical_data:
            return 0.7  # Default value
        
        # Simulate historical analysis
        topic = event["topic"].lower()
        
        # Count matching historical events
        matches = 0
        for historical_event in self.historical_data:
            historical_topic = historical_event.get("topic", "").lower()
            if topic in historical_topic or historical_topic in topic:
                matches += 1
        
        # Calculate factor based on matches
        if matches > 0:
            return min(1.0, 0.6 + (matches / len(self.historical_data)) * 0.4)
        else:
            return 0.6
    
    def _calculate_social_factor(self, event: Dict[str, Any]) -> float:
        """Calculate social amplification factor"""
        # In a real implementation, this would analyze social media dynamics
        # For this blueprint, we'll use social_volume as a proxy
        
        social_volume = event.get("social_volume", 0)
        
        # Calculate factor based on social volume
        return min(1.0, 0.5 + (social_volume / 50000) * 0.5)
    
    def _generate_recommendations(self, event: Dict[str, Any]) -> List[str]:
        """Generate recommended actions based on the event"""
        recommendations = []
        
        # Base recommendations
        recommendations.append("Monitor situation closely")
        
        # Add recommendations based on risk score
        risk_score = event["risk_score"]
        if risk_score > 0.9:
            recommendations.append("Immediate action required")
            recommendations.append("Activate emergency response protocols")
        elif risk_score > 0.8:
            recommendations.append("Prepare for potential emergency")
            recommendations.append("Alert relevant stakeholders")
        elif risk_score > 0.7:
            recommendations.append("Develop contingency plans")
            recommendations.append("Increase monitoring frequency")
        
        # Add recommendations based on impact areas
        impact_areas = event.get("potential_impact", [])
        for area in impact_areas:
            if area == "economic":
                recommendations.append("Assess financial exposure")
                recommendations.append("Review market positions")
            elif area == "social":
                recommendations.append("Prepare public communications")
                recommendations.append("Coordinate with community organizations")
            elif area == "political":
                recommendations.append("Engage with policy makers")
                recommendations.append("Monitor regulatory developments")
            elif area == "environmental":
                recommendations.append("Evaluate environmental impact")
                recommendations.append("Prepare containment measures")
        
        # Add recommendations based on timeframe
        timeframe = event.get("timeframe", "")
        if timeframe == "immediate":
            recommendations.append("Implement rapid response measures")
        elif timeframe == "short-term":
            recommendations.append("Prepare for near-term developments")
        elif timeframe == "long-term":
            recommendations.append("Develop strategic response plan")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create news fetcher
    news_fetcher = NewsFetcher()
    
    # Create API connection manager and social media manager
    api_manager = APIConnectionManager()
    social_manager = SocialMediaManager(api_manager)
    
    # Create trend analyzer
    trend_analyzer = TrendAnalyzer(news_fetcher, social_manager)
    
    # Create event predictor
    event_predictor = EventPredictor(trend_analyzer)
    
    # Create advanced algorithm
    algorithm = AdvancedAlgorithm(event_predictor)
    
    # Train the algorithm with sample data
    sample_data = [
        {
            "topic": "market crash",
            "risk_score": 0.85,
            "confidence": 0.75,
            "potential_impact": ["economic"],
            "timeframe": "short-term"
        },
        {
            "topic": "pandemic outbreak",
            "risk_score": 0.9,
            "confidence": 0.8,
            "potential_impact": ["social", "economic"],
            "timeframe": "immediate"
        }
    ]
    algorithm.train_model(sample_data)
    
    # Analyze and predict events
    predictions = algorithm.analyze_and_predict(timeframe="week", confidence_threshold=0.7)
    
    # Print results
    print(f"Prediction success: {predictions.get('success', False)}")
    print(f"Number of predictions: {len(predictions.get('predictions', []))}")
