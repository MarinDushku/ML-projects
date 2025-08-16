"""
Fantasy Premier League API Client

Comprehensive client for collecting data from the official FPL API and other sources.
Handles rate limiting, error handling, and data validation.
"""

import requests
import pandas as pd
import time
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLAPIClient:
    """Official Fantasy Premier League API client with rate limiting and caching."""
    
    def __init__(self, config_path: str = "config/api_endpoints.yaml"):
        """Initialize the FPL API client."""
        self.config = self._load_config(config_path)
        self.base_url = self.config['fpl_api']['base_url']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-AI/1.0 (Educational Project)'
        })
        self.last_request_time = 0
        self.rate_limit = self.config['rate_limits']['fpl_api']
        
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
            'fpl_api': {
                'base_url': 'https://fantasy.premierleague.com/api/',
                'endpoints': {
                    'bootstrap_static': 'bootstrap-static/',
                    'fixtures': 'fixtures/',
                    'gameweek_live': 'event/{}/live/',
                }
            },
            'rate_limits': {'fpl_api': 10}
        }
    
    def _rate_limit_wait(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make a rate-limited request to the FPL API."""
        self._rate_limit_wait()
        
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def get_bootstrap_static(self) -> Optional[Dict]:
        """
        Get bootstrap-static data containing teams, players, and gameweeks.
        This is the main endpoint with comprehensive FPL data.
        """
        endpoint = self.config['fpl_api']['endpoints']['bootstrap_static']
        return self._make_request(endpoint)
    
    def get_player_data(self) -> Optional[pd.DataFrame]:
        """Extract player data from bootstrap-static endpoint."""
        data = self.get_bootstrap_static()
        if not data:
            return None
            
        players_df = pd.DataFrame(data['elements'])
        
        # Add team and position information
        teams_df = pd.DataFrame(data['teams'])
        positions_df = pd.DataFrame(data['element_types'])
        
        # Merge team names
        players_df = players_df.merge(
            teams_df[['id', 'name', 'short_name']].rename(columns={
                'id': 'team', 'name': 'team_name', 'short_name': 'team_short'
            }),
            on='team'
        )
        
        # Merge position names
        players_df = players_df.merge(
            positions_df[['id', 'singular_name', 'singular_name_short']].rename(columns={
                'id': 'element_type', 'singular_name': 'position', 
                'singular_name_short': 'position_short'
            }),
            on='element_type'
        )
        
        return players_df
    
    def get_team_data(self) -> Optional[pd.DataFrame]:
        """Extract team data from bootstrap-static endpoint."""
        data = self.get_bootstrap_static()
        if not data:
            return None
        return pd.DataFrame(data['teams'])
    
    def get_gameweek_data(self) -> Optional[pd.DataFrame]:
        """Extract gameweek data from bootstrap-static endpoint."""
        data = self.get_bootstrap_static()
        if not data:
            return None
        return pd.DataFrame(data['events'])
    
    def get_fixtures(self) -> Optional[pd.DataFrame]:
        """Get fixture data for all gameweeks."""
        endpoint = self.config['fpl_api']['endpoints']['fixtures']
        data = self._make_request(endpoint)
        if not data:
            return None
        return pd.DataFrame(data)
    
    def get_gameweek_live(self, gameweek: int) -> Optional[Dict]:
        """Get live data for a specific gameweek."""
        endpoint = self.config['fpl_api']['endpoints']['gameweek_live'].format(gameweek)
        return self._make_request(endpoint)
    
    def get_player_gameweek_data(self, gameweek: int) -> Optional[pd.DataFrame]:
        """Extract player performance data for a specific gameweek."""
        data = self.get_gameweek_live(gameweek)
        if not data:
            return None
            
        player_data = []
        for element in data['elements']:
            stats = element['stats']
            stats['player_id'] = element['id']
            stats['gameweek'] = gameweek
            player_data.append(stats)
        
        return pd.DataFrame(player_data)
    
    def get_manager_data(self, manager_id: int) -> Optional[Dict]:
        """Get historical data for a specific manager."""
        endpoint = self.config['fpl_api']['endpoints']['manager_history'].format(manager_id)
        return self._make_request(endpoint)
    
    def get_manager_picks(self, manager_id: int, gameweek: int) -> Optional[Dict]:
        """Get team picks for a specific manager and gameweek."""
        endpoint = self.config['fpl_api']['endpoints']['manager_picks'].format(manager_id, gameweek)
        return self._make_request(endpoint)
    
    def save_data_to_csv(self, dataframe: pd.DataFrame, filename: str, 
                        directory: str = "data/raw/fpl_api"):
        """Save DataFrame to CSV with timestamp."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{directory}/{timestamp}_{filename}"
        dataframe.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath


class FPLDataCollector:
    """High-level data collector that orchestrates data collection from multiple sources."""
    
    def __init__(self, config_path: str = "config/api_endpoints.yaml"):
        """Initialize the data collector."""
        self.fpl_client = FPLAPIClient(config_path)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_current_season_data(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive current season data."""
        logger.info("Starting current season data collection...")
        
        collected_data = {}
        
        # Get player data
        players_df = self.fpl_client.get_player_data()
        if players_df is not None:
            collected_data['players'] = players_df
            self.fpl_client.save_data_to_csv(players_df, "players.csv")
            logger.info(f"Collected data for {len(players_df)} players")
        
        # Get team data
        teams_df = self.fpl_client.get_team_data()
        if teams_df is not None:
            collected_data['teams'] = teams_df
            self.fpl_client.save_data_to_csv(teams_df, "teams.csv")
            logger.info(f"Collected data for {len(teams_df)} teams")
        
        # Get gameweek data
        gameweeks_df = self.fpl_client.get_gameweek_data()
        if gameweeks_df is not None:
            collected_data['gameweeks'] = gameweeks_df
            self.fpl_client.save_data_to_csv(gameweeks_df, "gameweeks.csv")
            logger.info(f"Collected data for {len(gameweeks_df)} gameweeks")
        
        # Get fixtures
        fixtures_df = self.fpl_client.get_fixtures()
        if fixtures_df is not None:
            collected_data['fixtures'] = fixtures_df
            self.fpl_client.save_data_to_csv(fixtures_df, "fixtures.csv")
            logger.info(f"Collected {len(fixtures_df)} fixtures")
        
        logger.info("Current season data collection completed")
        return collected_data
    
    def collect_historical_gameweek_data(self, start_gameweek: int = 1, 
                                       end_gameweek: Optional[int] = None) -> pd.DataFrame:
        """Collect historical gameweek data for all completed gameweeks."""
        logger.info(f"Collecting historical gameweek data from GW{start_gameweek}...")
        
        if end_gameweek is None:
            # Get current gameweek from gameweeks data
            gameweeks_df = self.fpl_client.get_gameweek_data()
            if gameweeks_df is not None:
                current_gw = gameweeks_df[gameweeks_df['is_current'] == True]['id'].iloc[0]
                end_gameweek = current_gw - 1  # Only completed gameweeks
            else:
                end_gameweek = 10  # Default fallback
        
        all_gameweek_data = []
        
        for gw in range(start_gameweek, end_gameweek + 1):
            logger.info(f"Collecting data for gameweek {gw}")
            gw_data = self.fpl_client.get_player_gameweek_data(gw)
            if gw_data is not None:
                all_gameweek_data.append(gw_data)
            
            # Add small delay between requests
            time.sleep(1)
        
        if all_gameweek_data:
            combined_df = pd.concat(all_gameweek_data, ignore_index=True)
            self.fpl_client.save_data_to_csv(combined_df, "historical_gameweek_data.csv")
            logger.info(f"Collected historical data for gameweeks {start_gameweek}-{end_gameweek}")
            return combined_df
        
        return pd.DataFrame()
    
    def update_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Quick update of the latest available data."""
        logger.info("Updating latest FPL data...")
        return self.collect_current_season_data()


def main():
    """Main function to run data collection."""
    collector = FPLDataCollector()
    
    # Collect current season data
    current_data = collector.collect_current_season_data()
    
    # Collect historical gameweek data for completed gameweeks
    historical_data = collector.collect_historical_gameweek_data()
    
    logger.info("Data collection completed successfully!")
    logger.info(f"Collected data for {len(current_data)} data types")
    if not historical_data.empty:
        logger.info(f"Historical data contains {len(historical_data)} player-gameweek records")


if __name__ == "__main__":
    main()