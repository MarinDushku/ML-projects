"""
Goalkeeper-Specific FPL Prediction Model

Specialized model for predicting goalkeeper performance in Fantasy Premier League.
Focuses on clean sheets, save points, bonus points, and penalty saves.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.base_model import FPLBaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalkeeperPredictor(FPLBaseModel):
    """Specialized predictor for goalkeeper performance in FPL."""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """Initialize the goalkeeper predictor."""
        super().__init__("Goalkeeper", model_config)
        
        # Goalkeeper-specific configuration
        self.config.update({
            'clean_sheet_threshold': 0.3,  # Minimum probability for clean sheet prediction
            'save_points_per_save': 1/3,   # FPL scoring: 1 point per 3 saves
            'clean_sheet_points': 4,       # Points for a clean sheet
            'appearance_points': 2,        # Points for playing 60+ minutes
            'penalty_save_points': 5       # Points for saving a penalty
        })
    
    def _get_target_columns(self) -> List[str]:
        """Return target columns for goalkeeper prediction."""
        return [
            'total_points',
            'clean_sheets',
            'saves',
            'bonus',
            'minutes',
            'penalties_saved',
            'goals_conceded'
        ]
    
    def _get_position_features(self) -> List[str]:
        """Return the most important features for goalkeepers."""
        return [
            # Form metrics
            'form_points_5gw',
            'form_points_3gw',
            'form_minutes_5gw',
            'form_clean_sheets_5gw',
            'form_saves_5gw',
            'form_consistency_5gw',
            'weighted_form_5gw',
            
            # Clean sheet probability features
            'clean_sheet_probability',
            'average_saves_per_game',
            'penalty_save_rate',
            'save_points_potential',
            
            # Fixture features
            'fixture_difficulty',
            'is_home',
            'expected_goals_against',
            'next_5_fixtures_difficulty',
            
            # Team dynamics
            'price_rank_in_team',
            'is_key_player',
            'captain_potential',
            
            # Price and ownership
            'price',
            'price_per_point',
            'ownership_percentage',
            'differential_score',
            
            # Team defensive strength
            'expected_goals_for',  # Team's attacking strength affects opponent pressure
        ]
    
    def _calculate_fpl_points(self, predictions: pd.DataFrame) -> pd.Series:
        """
        Calculate total FPL points from component predictions for goalkeepers.
        
        FPL Goalkeeper Scoring:
        - Playing 60+ minutes: 2 points
        - Clean sheet: 4 points  
        - Every 3 saves: 1 point
        - Penalty save: 5 points
        - Bonus points: 1-3 points
        - Goal conceded: 0 points (no penalty)
        - Yellow card: -1 point
        - Red card: -3 points
        """
        total_points = pd.Series(0.0, index=predictions.index)
        
        # Appearance points (2 points for 60+ minutes)
        if 'pred_minutes' in predictions.columns:
            appearance_prob = np.clip(predictions['pred_minutes'] / 90.0, 0, 1)
            total_points += appearance_prob * self.config['appearance_points']
        
        # Clean sheet points (4 points)
        if 'pred_clean_sheets' in predictions.columns:
            clean_sheet_prob = np.clip(predictions['pred_clean_sheets'], 0, 1)
            total_points += clean_sheet_prob * self.config['clean_sheet_points']
        
        # Save points (1 point per 3 saves)
        if 'pred_saves' in predictions.columns:
            save_points = predictions['pred_saves'] * self.config['save_points_per_save']
            total_points += save_points
        
        # Penalty save points (5 points each)
        if 'pred_penalties_saved' in predictions.columns:
            penalty_points = predictions['pred_penalties_saved'] * self.config['penalty_save_points']
            total_points += penalty_points
        
        # Bonus points
        if 'pred_bonus' in predictions.columns:
            total_points += predictions['pred_bonus']
        
        # Ensure non-negative points
        total_points = np.maximum(total_points, 0)
        
        return total_points
    
    def _create_model(self, params: Dict, target: str):
        """Create specialized models for different goalkeeper targets."""
        
        # Use different models for different targets
        if target in ['clean_sheets']:
            # Classification-like approach for clean sheets
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 5),
                random_state=params.get('random_state', 42)
            )
        
        elif target in ['saves']:
            # Poisson-like regression for saves (count data)
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 120),
                learning_rate=params.get('learning_rate', 0.08),
                max_depth=params.get('max_depth', 4),
                random_state=params.get('random_state', 42),
                loss='poisson'  # Good for count data
            )
        
        elif target in ['bonus']:
            # Random Forest for bonus points (complex interactions)
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
        
        else:
            # Default XGBoost for other targets
            return super()._create_model(params, target)
    
    def predict_clean_sheet_probability(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict clean sheet probability with additional analysis.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with detailed clean sheet analysis
        """
        if 'clean_sheets' not in self.models:
            logger.warning("Clean sheet model not trained")
            return pd.DataFrame()
        
        predictions = self.predict(features_df)
        
        if predictions.empty:
            return pd.DataFrame()
        
        # Enhanced clean sheet analysis
        clean_sheet_analysis = predictions[['player_id', 'gameweek']].copy()
        
        # Base clean sheet probability
        if 'pred_clean_sheets' in predictions.columns:
            clean_sheet_analysis['clean_sheet_probability'] = predictions['pred_clean_sheets']
        
        # Add confidence intervals
        X, _ = self.prepare_data(features_df)
        if not X.empty:
            X_scaled = self.scalers['features'].transform(X)
            
            # Use multiple predictions to estimate uncertainty
            model = self.models['clean_sheets']
            if hasattr(model, 'predict_proba'):
                # For classification models
                proba = model.predict_proba(X_scaled)
                clean_sheet_analysis['clean_sheet_confidence'] = np.max(proba, axis=1)
            else:
                # For regression models, use prediction variance
                clean_sheet_analysis['clean_sheet_confidence'] = 0.7  # Default confidence
        
        # Risk factors
        clean_sheet_analysis['fixture_risk'] = features_df.loc[
            features_df['position'] == 'Goalkeeper', 'fixture_difficulty'
        ].values if 'fixture_difficulty' in features_df.columns else 3.0
        
        clean_sheet_analysis['away_penalty'] = ~features_df.loc[
            features_df['position'] == 'Goalkeeper', 'is_home'
        ].values if 'is_home' in features_df.columns else True
        
        # Expected goals against adjustment
        if 'expected_goals_against' in features_df.columns:
            clean_sheet_analysis['expected_goals_against'] = features_df.loc[
                features_df['position'] == 'Goalkeeper', 'expected_goals_against'
            ].values
            
            # Adjust clean sheet probability based on expected goals
            adj_factor = np.exp(-clean_sheet_analysis['expected_goals_against'])
            clean_sheet_analysis['adjusted_clean_sheet_prob'] = (
                clean_sheet_analysis['clean_sheet_probability'] * adj_factor
            )
        
        return clean_sheet_analysis
    
    def get_goalkeeper_insights(self, features_df: pd.DataFrame, 
                              player_id: int) -> Dict[str, any]:
        """
        Get detailed insights for a specific goalkeeper.
        
        Args:
            features_df: Feature DataFrame
            player_id: Goalkeeper ID
            
        Returns:
            Dictionary with goalkeeper insights
        """
        predictions = self.predict(features_df)
        player_pred = predictions[predictions['player_id'] == player_id]
        
        if player_pred.empty:
            return {}
        
        player_pred = player_pred.iloc[0]
        
        # Get player features
        player_features = features_df[features_df['player_id'] == player_id].iloc[0]
        
        insights = {
            'predicted_points': player_pred['predicted_points'],
            'clean_sheet_probability': player_pred.get('pred_clean_sheets', 0),
            'expected_saves': player_pred.get('pred_saves', 0),
            'save_points_potential': player_pred.get('pred_saves', 0) / 3,
            'bonus_potential': player_pred.get('pred_bonus', 0),
            
            # Risk assessment
            'fixture_difficulty': player_features.get('fixture_difficulty', 3),
            'is_home_fixture': player_features.get('is_home', 0.5),
            'expected_goals_against': player_features.get('expected_goals_against', 1.5),
            
            # Form analysis
            'recent_form': player_features.get('form_points_5gw', 0),
            'form_consistency': player_features.get('form_consistency_5gw', 0),
            
            # Value analysis
            'price': player_features.get('price', 5.0),
            'ownership': player_features.get('ownership_percentage', 10),
            'differential_potential': player_features.get('differential_score', 0.5),
            
            # Recommendations
            'start_recommendation': self._get_start_recommendation(player_pred, player_features),
            'captain_potential': self._get_captain_potential(player_pred, player_features),
            'transfer_priority': self._get_transfer_priority(player_pred, player_features)
        }
        
        return insights
    
    def _get_start_recommendation(self, predictions: pd.Series, 
                                features: pd.Series) -> str:
        """Get start/bench recommendation for goalkeeper."""
        predicted_points = predictions['predicted_points']
        
        if predicted_points >= 6:
            return "Strong Start - High clean sheet potential"
        elif predicted_points >= 4:
            return "Start - Decent fixtures"
        elif predicted_points >= 2:
            return "Risky - Consider alternatives"
        else:
            return "Bench - Poor fixture/form"
    
    def _get_captain_potential(self, predictions: pd.Series,
                             features: pd.Series) -> str:
        """Get captaincy recommendation for goalkeeper."""
        predicted_points = predictions['predicted_points']
        
        # Goalkeepers rarely make good captains, but in rare cases...
        if predicted_points >= 8 and features.get('differential_score', 0) > 0.8:
            return "Differential Captain - Very risky but high reward potential"
        else:
            return "Not Recommended - Better options available"
    
    def _get_transfer_priority(self, predictions: pd.Series,
                             features: pd.Series) -> str:
        """Get transfer priority recommendation."""
        predicted_points = predictions['predicted_points']
        price = features.get('price', 5.0)
        value_score = predicted_points / price
        
        if value_score >= 1.2:
            return "High Priority - Excellent value"
        elif value_score >= 1.0:
            return "Good Option - Solid value"
        elif value_score >= 0.8:
            return "Consider - Average value"
        else:
            return "Avoid - Poor value"


def main():
    """Example usage of the goalkeeper predictor."""
    predictor = GoalkeeperPredictor()
    logger.info("Goalkeeper predictor initialized")
    
    # Example of model configuration
    config = predictor.config
    logger.info(f"Goalkeeper model config: {config}")


if __name__ == "__main__":
    main()