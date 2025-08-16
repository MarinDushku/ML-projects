"""
Premier League Injury Data Scraper

Collects injury and availability data from multiple sources including
Premier Injuries website and API-Football service.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
import json
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryScraper:
    """Scraper for Premier League injury data from multiple sources."""
    
    def __init__(self, config_path: str = "config/api_endpoints.yaml"):
        """Initialize the injury scraper."""
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found."""
        return {
            'data_sources': {
                'premier_injuries': {
                    'base_url': 'https://www.premierinjuries.com/',
                    'injury_table': 'injury-table.php'
                }
            },
            'rate_limits': {'web_scraping': 2}
        }
    
    def _rate_limit_wait(self):
        """Ensure we don't overwhelm servers."""
        time.sleep(1.0 / self.config['rate_limits']['web_scraping'])
    
    def scrape_premier_injuries(self) -> Optional[pd.DataFrame]:
        """
        Scrape injury data from Premier Injuries website.
        This provides comprehensive injury status for all Premier League teams.
        """
        self._rate_limit_wait()
        
        base_url = self.config['data_sources']['premier_injuries']['base_url']
        injury_table = self.config['data_sources']['premier_injuries']['injury_table']
        url = f"{base_url}{injury_table}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the injury table
            injury_table = soup.find('table', class_='injury-table')
            if not injury_table:
                # Try alternative selectors
                injury_table = soup.find('table', id='injury-table')
                if not injury_table:
                    injury_table = soup.find('table')
            
            if not injury_table:
                logger.error("Could not find injury table on Premier Injuries page")
                return None
            
            # Extract injury data
            injuries = []
            rows = injury_table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:  # Ensure we have enough columns
                    injury_data = {
                        'player_name': cells[0].get_text(strip=True),
                        'team': cells[1].get_text(strip=True),
                        'injury_type': cells[2].get_text(strip=True),
                        'status': cells[3].get_text(strip=True),
                        'expected_return': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'premier_injuries'
                    }
                    injuries.append(injury_data)
            
            if injuries:
                df = pd.DataFrame(injuries)
                logger.info(f"Scraped {len(df)} injury records from Premier Injuries")
                return df
            else:
                logger.warning("No injury data found on Premier Injuries page")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Premier Injuries: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Premier Injuries data: {e}")
            return None
    
    def get_api_football_injuries(self, api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get injury data from API-Football service.
        Requires API key but has free tier available.
        """
        if not api_key:
            logger.info("No API-Football key provided, skipping API-Football injury data")
            return None
        
        self._rate_limit_wait()
        
        url = "https://v3.football.api-sports.io/injuries"
        headers = {
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': 'v3.football.api-sports.io'
        }
        
        # Premier League ID in API-Football is 39
        params = {
            'league': 39,
            'season': datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
        }
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('response'):
                injuries = []
                for injury in data['response']:
                    injury_data = {
                        'player_name': injury['player']['name'],
                        'team': injury['team']['name'],
                        'injury_type': injury['player']['reason'],
                        'status': 'Injured',
                        'expected_return': '',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'api_football'
                    }
                    injuries.append(injury_data)
                
                if injuries:
                    df = pd.DataFrame(injuries)
                    logger.info(f"Retrieved {len(df)} injury records from API-Football")
                    return df
            
            logger.warning("No injury data found from API-Football")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from API-Football: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing API-Football data: {e}")
            return None
    
    def scrape_official_premier_league(self) -> Optional[pd.DataFrame]:
        """
        Scrape injury data from official Premier League website.
        This provides the most up-to-date and official injury information.
        """
        self._rate_limit_wait()
        
        url = "https://www.premierleague.com/news"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury-related news articles
            injury_articles = []
            articles = soup.find_all('article') or soup.find_all('div', class_='news-article')
            
            for article in articles:
                title_elem = article.find(['h1', 'h2', 'h3']) or article.find(class_='title')
                if title_elem:
                    title = title_elem.get_text(strip=True).lower()
                    if any(keyword in title for keyword in ['injury', 'injured', 'doubt', 'fitness', 'unavailable']):
                        injury_articles.append({
                            'title': title_elem.get_text(strip=True),
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'premier_league_official'
                        })
            
            if injury_articles:
                df = pd.DataFrame(injury_articles)
                logger.info(f"Found {len(df)} injury-related articles from Premier League website")
                return df
            else:
                logger.info("No recent injury articles found on Premier League website")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Premier League website: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Premier League data: {e}")
            return None


class InjuryDataProcessor:
    """Processor for cleaning and standardizing injury data from multiple sources."""
    
    def __init__(self):
        """Initialize the injury data processor."""
        self.team_name_mapping = self._create_team_name_mapping()
        self.injury_severity_mapping = self._create_injury_severity_mapping()
    
    def _create_team_name_mapping(self) -> Dict[str, str]:
        """Create mapping for standardizing team names across sources."""
        return {
            # Common variations
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa', 
            'Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford',
            'Brighton': 'Brighton',
            'Brighton & Hove Albion': 'Brighton',
            'Chelsea': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton': 'Everton',
            'Fulham': 'Fulham',
            'Liverpool': 'Liverpool',
            'Luton': 'Luton',
            'Luton Town': 'Luton',
            'Man City': 'Manchester City',
            'Manchester City': 'Manchester City',
            'Man Utd': 'Manchester United',
            'Manchester United': 'Manchester United',
            'Newcastle': 'Newcastle',
            'Newcastle United': 'Newcastle',
            'Nott\'m Forest': 'Nottingham Forest',
            'Nottingham Forest': 'Nottingham Forest',
            'Sheffield Utd': 'Sheffield United',
            'Sheffield United': 'Sheffield United',
            'Tottenham': 'Tottenham',
            'West Ham': 'West Ham',
            'West Ham United': 'West Ham',
            'Wolves': 'Wolves',
            'Wolverhampton': 'Wolves'
        }
    
    def _create_injury_severity_mapping(self) -> Dict[str, int]:
        """Create mapping for injury severity scores."""
        return {
            'knock': 1,
            'minor': 1,
            'fatigue': 1,
            'slight': 1,
            'muscle': 2,
            'strain': 2,
            'sprain': 2,
            'bruise': 2,
            'fracture': 4,
            'break': 4,
            'torn': 4,
            'rupture': 5,
            'surgery': 5,
            'operation': 5,
            'long-term': 5
        }
    
    def standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team names across different sources."""
        if 'team' in df.columns:
            df['team_standardized'] = df['team'].map(self.team_name_mapping).fillna(df['team'])
        return df
    
    def calculate_injury_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate injury severity score based on injury type."""
        if 'injury_type' in df.columns:
            df['injury_severity'] = df['injury_type'].str.lower().map(
                lambda x: max([self.injury_severity_mapping.get(keyword, 2) 
                              for keyword in self.injury_severity_mapping.keys() 
                              if keyword in str(x)], default=2)
            )
        return df
    
    def estimate_return_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate return date based on injury type and status."""
        if 'expected_return' not in df.columns:
            df['expected_return'] = ''
        
        # Create estimated return date based on injury severity
        severity_to_weeks = {1: 1, 2: 3, 3: 6, 4: 10, 5: 20}
        
        if 'injury_severity' in df.columns:
            df['estimated_return_weeks'] = df['injury_severity'].map(severity_to_weeks)
            df['estimated_return_date'] = pd.to_datetime(datetime.now()) + pd.to_timedelta(
                df['estimated_return_weeks'] * 7, unit='D'
            )
        
        return df
    
    def process_injury_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all processing steps to injury data."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = self.standardize_team_names(df)
        df = self.calculate_injury_severity(df)
        df = self.estimate_return_date(df)
        
        # Add probability of availability for next gameweek
        if 'injury_severity' in df.columns:
            df['availability_probability'] = 1.0 - (df['injury_severity'] * 0.15)
            df['availability_probability'] = df['availability_probability'].clip(0, 1)
        
        return df


class InjuryDataCollector:
    """Main class for collecting and processing injury data from all sources."""
    
    def __init__(self, config_path: str = "config/api_endpoints.yaml"):
        """Initialize the injury data collector."""
        self.scraper = InjuryScraper(config_path)
        self.processor = InjuryDataProcessor()
        self.data_dir = Path("data/raw/injury_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_all_injury_data(self, api_key: Optional[str] = None) -> pd.DataFrame:
        """Collect injury data from all available sources."""
        logger.info("Starting comprehensive injury data collection...")
        
        all_injury_data = []
        
        # Collect from Premier Injuries website
        premier_injuries_data = self.scraper.scrape_premier_injuries()
        if premier_injuries_data is not None:
            all_injury_data.append(premier_injuries_data)
        
        # Collect from API-Football (if API key provided)
        if api_key:
            api_football_data = self.scraper.get_api_football_injuries(api_key)
            if api_football_data is not None:
                all_injury_data.append(api_football_data)
        
        # Collect from official Premier League website
        official_data = self.scraper.scrape_official_premier_league()
        if official_data is not None:
            all_injury_data.append(official_data)
        
        # Combine all data
        if all_injury_data:
            combined_df = pd.concat(all_injury_data, ignore_index=True)
            
            # Remove duplicates based on player name and team
            combined_df = combined_df.drop_duplicates(subset=['player_name', 'team'], keep='first')
            
            # Process the combined data
            processed_df = self.processor.process_injury_data(combined_df)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.data_dir / f"{timestamp}_injury_data.csv"
            processed_df.to_csv(filepath, index=False)
            
            logger.info(f"Collected and processed {len(processed_df)} injury records")
            logger.info(f"Data saved to {filepath}")
            
            return processed_df
        else:
            logger.warning("No injury data collected from any source")
            return pd.DataFrame()
    
    def get_latest_injury_data(self) -> pd.DataFrame:
        """Get the most recent injury data file."""
        csv_files = list(self.data_dir.glob("*_injury_data.csv"))
        if not csv_files:
            logger.warning("No injury data files found")
            return pd.DataFrame()
        
        # Get the most recent file
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading latest injury data from {latest_file}")
        return pd.read_csv(latest_file)


def main():
    """Main function to run injury data collection."""
    collector = InjuryDataCollector()
    
    # Collect injury data (add your API key here if you have one)
    injury_data = collector.collect_all_injury_data(api_key=None)
    
    if not injury_data.empty:
        print(f"Successfully collected {len(injury_data)} injury records")
        print("\nInjury Summary:")
        print(injury_data.groupby(['team_standardized', 'injury_severity']).size().unstack(fill_value=0))
    else:
        print("No injury data collected")


if __name__ == "__main__":
    main()