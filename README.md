# FPL-AI: Fantasy Premier League Intelligence System

An advanced machine learning system that predicts Fantasy Premier League player points and optimizes team selection using position-specific models, real-time data, and sophisticated feature engineering.

## 🎯 Overview

FPL-AI combines multiple ML techniques to solve the complex multi-objective optimization problem of Fantasy Premier League team selection. The system provides:

- **Position-specific point prediction models** for Goalkeepers, Defenders, Midfielders, and Forwards
- **Advanced fixture difficulty analysis** beyond basic FDR ratings
- **Injury risk assessment** and availability predictions
- **Meta-game intelligence** including ownership and price change predictions
- **Interactive dashboard** for team optimization and strategy planning

## 🚀 Key Features

### ML Models by Position
- **Goalkeepers**: Clean sheet probability + save points + bonus prediction
- **Defenders**: Clean sheet + attacking returns + bonus optimization
- **Midfielders**: Goals (5pts) + assists + clean sheet potential analysis
- **Forwards**: Goal scoring (4pts) + assist potential + bonus prediction

### Advanced Intelligence
- Dynamic team strength ratings using Elo system
- Player form analysis with weighted recent performance
- Injury probability modeling and recovery timeline estimation
- Transfer timing optimization and price change forecasting
- Captain selection recommendations with risk assessment

### Data Sources (All Free)
- Official FPL API for comprehensive player and match data
- GitHub FPL-Elo-Insights for advanced statistics and team ratings
- API-Football for injury data and player fitness status
- Premier Injuries website for latest injury updates
- FBref for detailed player performance statistics

## 📊 Technical Architecture

### Machine Learning Stack
- **Ensemble Methods**: XGBoost, LightGBM, CatBoost for robust predictions
- **Deep Learning**: Neural networks for complex pattern recognition
- **Time Series**: LSTM and Prophet for form and momentum analysis
- **Optimization**: Bayesian optimization for hyperparameter tuning
- **Interpretability**: SHAP and LIME for explainable predictions

### Data Pipeline
- Real-time data collection from multiple APIs
- Automated feature engineering with 50+ player metrics
- Model retraining with each gameweek
- Backtesting framework for strategy validation

## 🏗️ Project Structure

```
Premier_League_FPL/
├── src/
│   ├── data_collection/     # API clients and web scrapers
│   ├── feature_engineering/ # Advanced feature creation
│   ├── models/             # Position-specific ML models
│   ├── intelligence/       # Injury, rotation, and meta-game analysis
│   ├── optimization/       # Team selection and budget optimization
│   └── evaluation/         # Backtesting and performance metrics
├── notebooks/              # Jupyter notebooks for analysis
├── dashboard/              # Streamlit app and API
├── data/                   # Raw and processed datasets
└── models/                 # Trained model artifacts
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Configuration**
   ```bash
   cp config/api_endpoints.yaml.example config/api_endpoints.yaml
   # Edit with your API keys if needed
   ```

3. **Collect Initial Data**
   ```bash
   python src/data_collection/fpl_api_client.py
   ```

4. **Train Models**
   ```bash
   python src/models/train_all_models.py
   ```

5. **Launch Dashboard**
   ```bash
   streamlit run dashboard/streamlit_app/main.py
   ```

## 📈 Performance Metrics

The system tracks multiple performance indicators:
- **Point Prediction Accuracy**: RMSE and MAE for each position
- **Rank Improvement**: Historical rank performance vs baseline strategies
- **Transfer Efficiency**: ROI on transfer decisions
- **Captain Success Rate**: Above-average captain recommendations

## 🎮 Usage Examples

### Get Player Predictions
```python
from src.models.ensemble_predictor import FPLEnsemble

predictor = FPLEnsemble()
predictions = predictor.predict_gameweek(gameweek=15)
print(predictions.head())
```

### Optimize Team Selection
```python
from src.optimization.team_selector import TeamOptimizer

optimizer = TeamOptimizer(budget=100.0)
optimal_team = optimizer.select_team(predictions, gameweek=15)
```

### Analyze Transfer Opportunities
```python
from src.intelligence.transfer_advisor import TransferAdvisor

advisor = TransferAdvisor()
recommendations = advisor.get_transfer_recommendations(
    current_team=my_team,
    gameweek=15,
    transfers_available=1
)
```

## 🔧 Configuration

Key configuration files:
- `config/api_endpoints.yaml`: Data source endpoints
- `config/model_configs/`: ML model hyperparameters
- `config/feature_engineering.yaml`: Feature calculation settings

## 📊 Dashboard Features

The interactive Streamlit dashboard provides:
- **Player Analysis**: Deep dive into individual player metrics
- **Team Optimizer**: Visual team building with budget constraints
- **Transfer Planner**: Multi-gameweek transfer strategy
- **Captain Selector**: Advanced captaincy recommendations
- **Performance Tracker**: Historical strategy performance

## 🧪 Testing and Validation

- **Unit Tests**: Comprehensive testing for all modules
- **Integration Tests**: End-to-end pipeline validation
- **Backtesting**: Historical performance validation
- **Cross-validation**: Robust model evaluation

## 🚀 Deployment

The system supports multiple deployment options:
- **Local Development**: Streamlit dashboard
- **Docker Container**: Containerized deployment
- **Cloud Deployment**: AWS/GCP/Azure ready
- **API Service**: REST API for integration

## 📚 Documentation

Detailed documentation available in `docs/`:
- Technical methodology and model architecture
- API usage guides and examples
- FPL strategy insights and best practices

## 🤝 Contributing

This project showcases advanced ML engineering skills applied to Fantasy Premier League optimization. Key technical achievements:

1. **Multi-modal Learning**: Combining tabular, time series, and text data
2. **Real-time Inference**: Sub-minute prediction updates
3. **Domain Expertise**: Deep FPL knowledge encoded in features
4. **Production Ready**: Comprehensive testing, monitoring, and deployment

## 📄 License

This project is for educational and portfolio demonstration purposes.

---

*Built with Python, scikit-learn, XGBoost, TensorFlow, and Streamlit*