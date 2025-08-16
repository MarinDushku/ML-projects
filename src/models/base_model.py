"""
Base Model Class for FPL Position-Specific Predictions

Provides common functionality for all position-specific models including
training, prediction, evaluation, and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import logging
from datetime import datetime
from abc import ABC, abstractmethod

import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLBaseModel(ABC):
    """Abstract base class for all FPL position-specific models."""
    
    def __init__(self, position: str, model_config: Optional[Dict] = None):
        """
        Initialize the base model.
        
        Args:
            position: Player position (Goalkeeper, Defender, Midfielder, Forward)
            model_config: Model configuration parameters
        """
        self.position = position
        self.config = model_config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.model_metadata = {}
        
        # Create model save directory
        self.model_dir = Path(f"models/trained_models/{position.lower()}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for the model."""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'n_splits': 5,
            'test_size': 0.2,
            'optuna_trials': 100
        }
    
    @abstractmethod
    def _get_target_columns(self) -> List[str]:
        """Return the target columns for this position."""
        pass
    
    @abstractmethod
    def _get_position_features(self) -> List[str]:
        """Return the most important features for this position."""
        pass
    
    @abstractmethod
    def _calculate_fpl_points(self, predictions: pd.DataFrame) -> pd.Series:
        """Calculate FPL points from component predictions."""
        pass
    
    def prepare_data(self, features_df: pd.DataFrame, 
                    targets_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare data for training or prediction.
        
        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame (None for prediction)
            
        Returns:
            Prepared features and targets
        """
        # Filter by position
        position_data = features_df[features_df['position'] == self.position].copy()
        
        if position_data.empty:
            logger.warning(f"No data found for position {self.position}")
            return pd.DataFrame(), None
        
        # Select relevant features
        feature_cols = self._get_position_features()
        available_features = [col for col in feature_cols if col in position_data.columns]
        
        if len(available_features) < len(feature_cols) * 0.7:
            logger.warning(f"Only {len(available_features)}/{len(feature_cols)} features available")
        
        X = position_data[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Prepare targets if provided
        y = None
        if targets_df is not None:
            target_cols = self._get_target_columns()
            available_targets = [col for col in target_cols if col in targets_df.columns]
            
            # Merge targets with position data
            merged_data = position_data[['player_id', 'gameweek']].merge(
                targets_df[['player_id', 'gameweek'] + available_targets],
                on=['player_id', 'gameweek'],
                how='inner'
            )
            
            if not merged_data.empty:
                y = merged_data[available_targets]
                # Align X with merged data
                X = X.loc[X.index.isin(merged_data.index)]
        
        return X, y
    
    def train(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
              optimize_hyperparams: bool = True) -> Dict[str, float]:
        """
        Train the model(s) for this position.
        
        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.position} model...")
        
        X, y = self.prepare_data(features_df, targets_df)
        
        if X.empty or y is None or y.empty:
            logger.error(f"No training data available for {self.position}")
            return {}
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.scalers['features'] = scaler
        
        metrics = {}
        target_columns = y.columns
        
        # Train a model for each target
        for target in target_columns:
            logger.info(f"Training model for {target}")
            
            # Remove rows with missing targets
            valid_mask = ~y[target].isna()
            X_target = X_scaled[valid_mask]
            y_target = y[target][valid_mask]
            
            if len(y_target) < 50:
                logger.warning(f"Insufficient data for {target}: {len(y_target)} samples")
                continue
            
            # Optimize hyperparameters if requested
            if optimize_hyperparams:
                best_params = self._optimize_hyperparameters(X_target, y_target, target)
            else:
                best_params = self.config.copy()
            
            # Train final model
            model = self._create_model(best_params, target)
            model.fit(X_target, y_target)
            
            # Store model and evaluate
            self.models[target] = model
            
            # Cross-validation metrics
            cv_scores = self._cross_validate(model, X_target, y_target)
            metrics[f'{target}_cv_rmse'] = cv_scores['rmse']
            metrics[f'{target}_cv_mae'] = cv_scores['mae']
            metrics[f'{target}_cv_r2'] = cv_scores['r2']
            
            # Feature importance
            self.feature_importance[target] = self._get_feature_importance(model, X.columns)
        
        self.is_trained = True
        self.model_metadata = {
            'position': self.position,
            'n_features': len(X.columns),
            'n_samples': len(X),
            'target_columns': list(target_columns),
            'training_date': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        logger.info(f"Training completed for {self.position}. Metrics: {metrics}")
        return metrics
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for this position.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.position} model has not been trained")
        
        X, _ = self.prepare_data(features_df)
        
        if X.empty:
            return pd.DataFrame()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = {}
        
        # Make predictions for each target
        for target, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[f'pred_{target}'] = pred
        
        pred_df = pd.DataFrame(predictions, index=X.index)
        
        # Add player identifiers
        position_data = features_df[features_df['position'] == self.position]
        pred_df['player_id'] = position_data.loc[X.index, 'player_id']
        pred_df['gameweek'] = position_data.loc[X.index, 'gameweek'] 
        pred_df['position'] = self.position
        
        # Calculate total FPL points
        pred_df['predicted_points'] = self._calculate_fpl_points(pred_df)
        
        return pred_df
    
    def _create_model(self, params: Dict, target: str):
        """Create model instance based on configuration."""
        model_type = params.get('model_type', 'xgboost')
        
        if model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=params.get('random_state', 42),
                n_jobs=-1,
                verbosity=-1
            )
        elif model_type == 'catboost':
            return CatBoostRegressor(
                iterations=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),
                random_seed=params.get('random_state', 42),
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, target: str) -> Dict:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Optimizing hyperparameters for {target}")
        
        def objective(trial):
            params = {
                'model_type': trial.suggest_categorical('model_type', ['xgboost', 'lightgbm']),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': 42
            }
            
            model = self._create_model(params, target)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = mean_squared_error(y_val, pred, squared=False)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.get('optuna_trials', 50))
        
        return study.best_params
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation and return metrics."""
        tscv = TimeSeriesSplit(n_splits=self.config.get('n_splits', 5))
        
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            rmse_scores.append(mean_squared_error(y_val, pred, squared=False))
            mae_scores.append(mean_absolute_error(y_val, pred))
            r2_scores.append(r2_score(y_val, pred))
        
        return {
            'rmse': np.mean(rmse_scores),
            'mae': np.mean(mae_scores),
            'r2': np.mean(r2_scores)
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the model."""
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_)
        else:
            return {}
        
        importance_dict = dict(zip(feature_names, importance_scores))
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_importance(self, target: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all targets or a specific target."""
        if target:
            return {target: self.feature_importance.get(target, {})}
        return self.feature_importance
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.model_dir / f"{self.position.lower()}_model_{timestamp}.joblib"
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'metadata': self.model_metadata,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.model_metadata = model_data['metadata']
        self.config = model_data.get('config', self.config)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def explain_prediction(self, features_df: pd.DataFrame, 
                         player_id: int, target: str) -> Dict[str, Any]:
        """
        Explain a specific prediction using SHAP.
        
        Args:
            features_df: Feature DataFrame
            player_id: Player ID to explain
            target: Target variable to explain
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        if target not in self.models:
            raise ValueError(f"Model for {target} not found")
        
        X, _ = self.prepare_data(features_df)
        player_data = features_df[features_df['player_id'] == player_id]
        
        if player_data.empty:
            raise ValueError(f"Player {player_id} not found in data")
        
        # Get player's features
        player_features = X.loc[X.index.isin(player_data.index)].iloc[0:1]
        X_scaled = self.scalers['features'].transform(player_features)
        
        # Create SHAP explainer
        model = self.models[target]
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)
        
        return {
            'shap_values': shap_values.values[0],
            'feature_names': player_features.columns.tolist(),
            'prediction': model.predict(X_scaled)[0],
            'base_value': shap_values.base_values[0]
        }


def create_custom_fpl_loss(position: str):
    """Create custom loss function that matches FPL scoring system."""
    
    def fpl_loss(y_true, y_pred):
        """Custom loss function that penalizes errors based on FPL point values."""
        # Different positions have different point values
        if position == 'Goalkeeper':
            # Clean sheets worth 4 points, saves worth 1/3 point each
            clean_sheet_weight = 4.0
            saves_weight = 1/3
        elif position == 'Defender':
            # Clean sheets worth 4 points, goals worth 6 points
            clean_sheet_weight = 4.0
            goal_weight = 6.0
        elif position == 'Midfielder':
            # Goals worth 5 points, assists worth 3 points
            goal_weight = 5.0
            assist_weight = 3.0
        else:  # Forward
            # Goals worth 4 points, assists worth 3 points
            goal_weight = 4.0
            assist_weight = 3.0
        
        # Calculate weighted MSE based on point values
        error = y_true - y_pred
        weighted_error = error ** 2
        
        return np.mean(weighted_error)
    
    return fpl_loss