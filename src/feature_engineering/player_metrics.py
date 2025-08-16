"""
Advanced Player Feature Engineering for FPL Prediction

Creates sophisticated features for each player position including form analysis,
fixture difficulty, team dynamics, and position-specific metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerMetricsEngine:
    """Advanced feature engineering for FPL player prediction."""
    
    def __init__(self):
        """Initialize the player metrics engine."""
        self.position_weights = self._get_position_weights()
        self.form_windows = [3, 5, 8, 10]  # Different form calculation windows
        self.scalers = {}
        
    def _get_position_weights(self) -> Dict[str, Dict[str, float]]:
        """Define position-specific weights for different metrics."""
        return {
            'Goalkeeper': {
                'clean_sheet_weight': 1.0,
                'saves_weight': 0.8,
                'goals_weight': 0.1,
                'assists_weight': 0.1,
                'bonus_weight': 0.6,
                'minutes_weight': 1.0
            },
            'Defender': {
                'clean_sheet_weight': 0.9,
                'goals_weight': 0.7,
                'assists_weight': 0.6,
                'bonus_weight': 0.7,
                'minutes_weight': 0.9,
                'threat_weight': 0.5
            },
            'Midfielder': {
                'goals_weight': 0.9,
                'assists_weight': 0.9,
                'clean_sheet_weight': 0.3,
                'bonus_weight': 0.8,
                'minutes_weight': 0.8,
                'creativity_weight': 0.9
            },
            'Forward': {
                'goals_weight': 1.0,
                'assists_weight': 0.7,
                'bonus_weight': 0.8,
                'minutes_weight': 0.8,
                'threat_weight': 1.0,
                'penalties_weight': 0.9
            }
        }
    
    def calculate_form_metrics(self, df: pd.DataFrame, player_id: int, 
                             current_gameweek: int) -> Dict[str, float]:
        """
        Calculate various form metrics for a player.
        
        Args:
            df: Historical gameweek data
            player_id: FPL player ID
            current_gameweek: Current gameweek number
            
        Returns:
            Dictionary of form metrics
        """
        player_data = df[df['player_id'] == player_id].copy()
        
        if player_data.empty:
            return self._get_default_form_metrics()
        
        # Sort by gameweek
        player_data = player_data.sort_values('gameweek')
        
        form_metrics = {}
        
        # Calculate form for different windows
        for window in self.form_windows:
            recent_data = player_data[
                player_data['gameweek'] >= (current_gameweek - window)
            ].copy()
            
            if not recent_data.empty:
                # Basic form metrics
                form_metrics[f'form_points_{window}gw'] = recent_data['total_points'].mean()
                form_metrics[f'form_minutes_{window}gw'] = recent_data['minutes'].mean()
                form_metrics[f'form_goals_{window}gw'] = recent_data['goals_scored'].sum()
                form_metrics[f'form_assists_{window}gw'] = recent_data['assists'].sum()
                form_metrics[f'form_bonus_{window}gw'] = recent_data['bonus'].mean()
                
                # Advanced form metrics
                form_metrics[f'form_consistency_{window}gw'] = (
                    1.0 / (1.0 + recent_data['total_points'].std()) if len(recent_data) > 1 else 1.0
                )
                form_metrics[f'form_trend_{window}gw'] = self._calculate_trend(
                    recent_data['total_points'].values
                )
                
                # Position-specific metrics
                if 'clean_sheets' in recent_data.columns:
                    form_metrics[f'form_clean_sheets_{window}gw'] = recent_data['clean_sheets'].sum()
                if 'saves' in recent_data.columns:
                    form_metrics[f'form_saves_{window}gw'] = recent_data['saves'].sum()
        
        # Calculate weighted recent form (more weight on recent games)
        if len(player_data) >= 5:
            weights = np.exp(np.linspace(-1, 0, min(5, len(player_data))))
            weights = weights / weights.sum()
            recent_5 = player_data.tail(5)
            form_metrics['weighted_form_5gw'] = np.average(recent_5['total_points'], weights=weights)
        
        return form_metrics
    
    def _get_default_form_metrics(self) -> Dict[str, float]:
        """Return default form metrics for players with no history."""
        defaults = {}
        for window in self.form_windows:
            defaults.update({
                f'form_points_{window}gw': 0.0,
                f'form_minutes_{window}gw': 0.0,
                f'form_goals_{window}gw': 0.0,
                f'form_assists_{window}gw': 0.0,
                f'form_bonus_{window}gw': 0.0,
                f'form_consistency_{window}gw': 0.0,
                f'form_trend_{window}gw': 0.0
            })
        defaults['weighted_form_5gw'] = 0.0
        return defaults
    
    def _calculate_trend(self, points_series: np.ndarray) -> float:
        """Calculate the trend in points over recent games."""
        if len(points_series) < 2:
            return 0.0
        
        x = np.arange(len(points_series))
        trend = np.polyfit(x, points_series, 1)[0]
        return float(trend)
    
    def calculate_fixture_features(self, fixtures_df: pd.DataFrame, 
                                 team_id: int, gameweek: int,
                                 team_strength_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate fixture-related features for upcoming games.
        
        Args:
            fixtures_df: Fixture data
            team_id: Team ID
            gameweek: Target gameweek
            team_strength_df: Team strength ratings
            
        Returns:
            Dictionary of fixture features
        """
        # Get upcoming fixtures for the team
        upcoming_fixtures = fixtures_df[
            (fixtures_df['event'] == gameweek) & 
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
        ].copy()
        
        if upcoming_fixtures.empty:
            return self._get_default_fixture_features()
        
        fixture_features = {}
        
        for _, fixture in upcoming_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            
            # Basic fixture info
            fixture_features['is_home'] = float(is_home)
            fixture_features['opponent_id'] = opponent_id
            
            # Fixture difficulty (simple version)
            if team_strength_df is not None and not team_strength_df.empty:
                opponent_strength = team_strength_df[
                    team_strength_df['team_id'] == opponent_id
                ]['strength'].iloc[0] if len(team_strength_df[
                    team_strength_df['team_id'] == opponent_id
                ]) > 0 else 3.0
                
                # Adjust for home advantage
                if is_home:
                    fixture_features['fixture_difficulty'] = max(1.0, opponent_strength - 0.5)
                else:
                    fixture_features['fixture_difficulty'] = min(5.0, opponent_strength + 0.5)
            else:
                fixture_features['fixture_difficulty'] = 3.0  # Default neutral difficulty
            
            # Calculate expected goals based on team performance
            fixture_features['expected_goals_for'] = self._calculate_expected_goals(
                team_id, opponent_id, is_home, team_strength_df
            )
            fixture_features['expected_goals_against'] = self._calculate_expected_goals(
                opponent_id, team_id, not is_home, team_strength_df
            )
        
        # Calculate next 5 fixtures difficulty
        next_5_fixtures = fixtures_df[
            (fixtures_df['event'].between(gameweek, gameweek + 4)) &
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
        ]
        
        if not next_5_fixtures.empty and team_strength_df is not None:
            difficulties = []
            for _, fix in next_5_fixtures.iterrows():
                home = fix['team_h'] == team_id
                opp = fix['team_a'] if home else fix['team_h']
                opp_strength = team_strength_df[
                    team_strength_df['team_id'] == opp
                ]['strength'].iloc[0] if len(team_strength_df[
                    team_strength_df['team_id'] == opp
                ]) > 0 else 3.0
                difficulties.append(opp_strength - 0.3 if home else opp_strength + 0.3)
            
            fixture_features['next_5_fixtures_difficulty'] = np.mean(difficulties)
        else:
            fixture_features['next_5_fixtures_difficulty'] = 3.0
        
        return fixture_features
    
    def _get_default_fixture_features(self) -> Dict[str, float]:
        """Return default fixture features."""
        return {
            'is_home': 0.5,
            'opponent_id': 0,
            'fixture_difficulty': 3.0,
            'expected_goals_for': 1.5,
            'expected_goals_against': 1.5,
            'next_5_fixtures_difficulty': 3.0
        }
    
    def _calculate_expected_goals(self, team_id: int, opponent_id: int, 
                                is_home: bool, team_strength_df: Optional[pd.DataFrame]) -> float:
        """Calculate expected goals for a team in a fixture."""
        if team_strength_df is None or team_strength_df.empty:
            return 1.5  # Default
        
        # Get team attacking and opponent defensive strength
        team_data = team_strength_df[team_strength_df['team_id'] == team_id]
        opponent_data = team_strength_df[team_strength_df['team_id'] == opponent_id]
        
        if team_data.empty or opponent_data.empty:
            return 1.5
        
        team_attack = team_data['attack_strength'].iloc[0] if 'attack_strength' in team_data.columns else 1.0
        opponent_defense = opponent_data['defense_strength'].iloc[0] if 'defense_strength' in opponent_data.columns else 1.0
        
        # Base expected goals calculation
        expected_goals = team_attack / opponent_defense * 1.3  # League average
        
        # Home advantage
        if is_home:
            expected_goals *= 1.15
        
        return max(0.5, min(4.0, expected_goals))
    
    def calculate_team_dynamics_features(self, players_df: pd.DataFrame, 
                                       team_id: int, player_id: int) -> Dict[str, float]:
        """
        Calculate features related to team dynamics and player role.
        
        Args:
            players_df: Current player data
            team_id: Team ID
            player_id: Player ID
            
        Returns:
            Dictionary of team dynamics features
        """
        team_players = players_df[players_df['team'] == team_id].copy()
        player_data = players_df[players_df['id'] == player_id].iloc[0]
        
        dynamics_features = {}
        
        # Player's role in team (based on price and points)
        if not team_players.empty:
            team_price_rank = (team_players['now_cost'] >= player_data['now_cost']).sum()
            team_points_rank = (team_players['total_points'] >= player_data['total_points']).sum()
            
            dynamics_features['price_rank_in_team'] = team_price_rank / len(team_players)
            dynamics_features['points_rank_in_team'] = team_points_rank / len(team_players)
            
            # Key player indicator
            dynamics_features['is_key_player'] = float(
                (team_price_rank <= 3) or (team_points_rank <= 3)
            )
            
            # Position competition
            position_players = team_players[
                team_players['element_type'] == player_data['element_type']
            ]
            if len(position_players) > 1:
                pos_rank = (position_players['total_points'] >= player_data['total_points']).sum()
                dynamics_features['position_competition'] = pos_rank / len(position_players)
            else:
                dynamics_features['position_competition'] = 1.0
        
        # Set piece taker likelihood (simple heuristic)
        dynamics_features['set_piece_likelihood'] = self._estimate_set_piece_likelihood(
            player_data, team_players
        )
        
        # Penalty taker likelihood
        dynamics_features['penalty_likelihood'] = self._estimate_penalty_likelihood(
            player_data, team_players
        )
        
        # Captain potential (based on ownership and points)
        dynamics_features['captain_potential'] = (
            player_data.get('selected_by_percent', 0) * 
            player_data.get('total_points', 0) / 100
        )
        
        return dynamics_features
    
    def _estimate_set_piece_likelihood(self, player_data: pd.Series, 
                                     team_players: pd.DataFrame) -> float:
        """Estimate likelihood of being a set piece taker."""
        # Simple heuristic based on position and creativity stats
        position = player_data.get('element_type', 3)
        
        if position == 1:  # Goalkeeper
            return 0.0
        elif position == 2:  # Defender
            return 0.3 if player_data.get('total_points', 0) > 50 else 0.1
        elif position == 3:  # Midfielder
            return 0.7 if player_data.get('creativity', 0) > 50 else 0.4
        else:  # Forward
            return 0.2
    
    def _estimate_penalty_likelihood(self, player_data: pd.Series,
                                   team_players: pd.DataFrame) -> float:
        """Estimate likelihood of being a penalty taker."""
        # Based on position, price, and previous penalty record
        position = player_data.get('element_type', 3)
        price = player_data.get('now_cost', 50)
        
        if position == 1:  # Goalkeeper
            return 0.0
        elif position in [3, 4]:  # Midfielder or Forward
            # Higher chance for expensive attacking players
            base_likelihood = 0.3 if price > 80 else 0.1
            
            # Boost for high scorers
            if player_data.get('goals_scored', 0) > 5:
                base_likelihood += 0.2
                
            return min(0.8, base_likelihood)
        else:  # Defender
            return 0.05
    
    def calculate_position_specific_features(self, player_data: pd.Series,
                                           historical_df: pd.DataFrame,
                                           position: str) -> Dict[str, float]:
        """
        Calculate position-specific advanced features.
        
        Args:
            player_data: Current player data
            historical_df: Historical gameweek data
            position: Player position (Goalkeeper, Defender, Midfielder, Forward)
            
        Returns:
            Dictionary of position-specific features
        """
        features = {}
        player_id = player_data['id']
        
        # Get player's historical data
        player_history = historical_df[historical_df['player_id'] == player_id].copy()
        
        if position == 'Goalkeeper':
            features.update(self._goalkeeper_features(player_data, player_history))
        elif position == 'Defender':
            features.update(self._defender_features(player_data, player_history))
        elif position == 'Midfielder':
            features.update(self._midfielder_features(player_data, player_history))
        elif position == 'Forward':
            features.update(self._forward_features(player_data, player_history))
        
        return features
    
    def _goalkeeper_features(self, player_data: pd.Series, 
                           player_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate goalkeeper-specific features."""
        features = {}
        
        # Clean sheet probability based on historical performance
        if not player_history.empty:
            clean_sheet_rate = player_history['clean_sheets'].sum() / len(player_history)
            save_rate = player_history['saves'].mean()
            penalty_save_rate = player_history.get('penalties_saved', pd.Series([0])).sum() / max(1, len(player_history))
        else:
            clean_sheet_rate = 0.3
            save_rate = 2.0
            penalty_save_rate = 0.1
        
        features.update({
            'clean_sheet_probability': clean_sheet_rate,
            'average_saves_per_game': save_rate,
            'penalty_save_rate': penalty_save_rate,
            'save_points_potential': save_rate / 3.0,  # 1 point per 3 saves
        })
        
        return features
    
    def _defender_features(self, player_data: pd.Series,
                         player_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate defender-specific features."""
        features = {}
        
        if not player_history.empty:
            clean_sheet_rate = player_history['clean_sheets'].sum() / len(player_history)
            goal_rate = player_history['goals_scored'].sum() / len(player_history)
            assist_rate = player_history['assists'].sum() / len(player_history)
            threat_level = player_history.get('threat', pd.Series([0])).mean()
        else:
            clean_sheet_rate = 0.3
            goal_rate = 0.1
            assist_rate = 0.05
            threat_level = 10.0
        
        features.update({
            'clean_sheet_probability': clean_sheet_rate,
            'goal_scoring_rate': goal_rate,
            'assist_rate': assist_rate,
            'attacking_threat': threat_level,
            'defensive_attacking_balance': (threat_level + clean_sheet_rate * 50) / 2,
        })
        
        return features
    
    def _midfielder_features(self, player_data: pd.Series,
                           player_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate midfielder-specific features."""
        features = {}
        
        if not player_history.empty:
            goal_rate = player_history['goals_scored'].sum() / len(player_history)
            assist_rate = player_history['assists'].sum() / len(player_history)
            creativity = player_history.get('creativity', pd.Series([0])).mean()
            influence = player_history.get('influence', pd.Series([0])).mean()
            clean_sheet_involvement = player_history['clean_sheets'].sum() / len(player_history)
        else:
            goal_rate = 0.15
            assist_rate = 0.1
            creativity = 30.0
            influence = 50.0
            clean_sheet_involvement = 0.2
        
        features.update({
            'goal_scoring_rate': goal_rate,
            'assist_rate': assist_rate,
            'creativity_index': creativity,
            'influence_index': influence,
            'attacking_returns_rate': goal_rate + assist_rate,
            'clean_sheet_involvement': clean_sheet_involvement,
        })
        
        return features
    
    def _forward_features(self, player_data: pd.Series,
                        player_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate forward-specific features."""
        features = {}
        
        if not player_history.empty:
            goal_rate = player_history['goals_scored'].sum() / len(player_history)
            assist_rate = player_history['assists'].sum() / len(player_history)
            threat_level = player_history.get('threat', pd.Series([0])).mean()
            ict_index = player_history.get('ict_index', pd.Series([0])).mean()
        else:
            goal_rate = 0.3
            assist_rate = 0.1
            threat_level = 50.0
            ict_index = 60.0
        
        features.update({
            'goal_scoring_rate': goal_rate,
            'assist_rate': assist_rate,
            'attacking_threat': threat_level,
            'ict_index': ict_index,
            'attacking_returns_rate': goal_rate + assist_rate,
            'minutes_per_goal': 90 / max(0.01, goal_rate),
        })
        
        return features
    
    def calculate_price_and_ownership_features(self, player_data: pd.Series,
                                             players_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features related to price and ownership."""
        features = {}
        
        # Price features
        price = player_data.get('now_cost', 50) / 10.0  # Convert to actual price
        features['price'] = price
        features['price_per_point'] = price / max(1, player_data.get('total_points', 1))
        
        # Value features compared to position
        position_players = players_df[
            players_df['element_type'] == player_data['element_type']
        ]
        
        if not position_players.empty:
            price_percentile = (position_players['now_cost'] <= player_data['now_cost']).mean()
            points_percentile = (position_players['total_points'] <= player_data['total_points']).mean()
            
            features['price_percentile_in_position'] = price_percentile
            features['points_percentile_in_position'] = points_percentile
            features['value_score'] = points_percentile - price_percentile
        
        # Ownership features
        ownership = player_data.get('selected_by_percent', 5.0)
        features['ownership_percentage'] = ownership
        features['differential_score'] = max(0, 50 - ownership) / 50  # Lower for highly owned
        
        return features


class FeatureEngineeringPipeline:
    """Main pipeline for comprehensive feature engineering."""
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.metrics_engine = PlayerMetricsEngine()
        self.feature_columns = []
        
    def create_features_for_gameweek(self, 
                                   players_df: pd.DataFrame,
                                   historical_df: pd.DataFrame,
                                   fixtures_df: pd.DataFrame,
                                   gameweek: int,
                                   team_strength_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create comprehensive features for all players for a specific gameweek.
        
        Args:
            players_df: Current player data
            historical_df: Historical gameweek data
            fixtures_df: Fixture data
            gameweek: Target gameweek
            team_strength_df: Team strength ratings
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Creating features for gameweek {gameweek}")
        
        all_features = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            team_id = player['team']
            position = player.get('position', 'Midfielder')
            
            # Initialize feature dictionary
            features = {
                'player_id': player_id,
                'gameweek': gameweek,
                'position': position,
                'team_id': team_id,
            }
            
            # Form metrics
            form_features = self.metrics_engine.calculate_form_metrics(
                historical_df, player_id, gameweek
            )
            features.update(form_features)
            
            # Fixture features
            fixture_features = self.metrics_engine.calculate_fixture_features(
                fixtures_df, team_id, gameweek, team_strength_df
            )
            features.update(fixture_features)
            
            # Team dynamics
            team_features = self.metrics_engine.calculate_team_dynamics_features(
                players_df, team_id, player_id
            )
            features.update(team_features)
            
            # Position-specific features
            position_features = self.metrics_engine.calculate_position_specific_features(
                player, historical_df, position
            )
            features.update(position_features)
            
            # Price and ownership
            price_features = self.metrics_engine.calculate_price_and_ownership_features(
                player, players_df
            )
            features.update(price_features)
            
            all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in features_df.columns 
                              if col not in ['player_id', 'gameweek', 'position', 'team_id']]
        
        logger.info(f"Created {len(self.feature_columns)} features for {len(features_df)} players")
        
        return features_df
    
    def get_feature_importance_by_position(self) -> Dict[str, List[str]]:
        """Return the most important features for each position."""
        return {
            'Goalkeeper': [
                'clean_sheet_probability', 'average_saves_per_game', 'form_points_5gw',
                'fixture_difficulty', 'is_home', 'minutes_weight'
            ],
            'Defender': [
                'clean_sheet_probability', 'attacking_threat', 'form_points_5gw',
                'fixture_difficulty', 'is_home', 'goal_scoring_rate'
            ],
            'Midfielder': [
                'goal_scoring_rate', 'assist_rate', 'creativity_index', 'form_points_5gw',
                'attacking_returns_rate', 'set_piece_likelihood'
            ],
            'Forward': [
                'goal_scoring_rate', 'attacking_threat', 'form_points_5gw',
                'ict_index', 'attacking_returns_rate', 'penalty_likelihood'
            ]
        }


def main():
    """Example usage of the feature engineering pipeline."""
    # This would typically be called with real data
    logger.info("Feature engineering pipeline ready for use")
    
    pipeline = FeatureEngineeringPipeline()
    feature_importance = pipeline.get_feature_importance_by_position()
    
    for position, features in feature_importance.items():
        logger.info(f"{position} key features: {features[:3]}")


if __name__ == "__main__":
    main()