"""
Manual Injury Data Helper

Utilities for processing manually collected injury data from Sky Sports, BBC Sport, etc.
Helps format and validate injury data for integration with FPL predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional


class ManualInjuryProcessor:
    """Helper class for processing manually collected injury data."""
    
    def __init__(self):
        """Initialize the manual injury processor."""
        self.team_name_mapping = self._create_team_name_mapping()
        self.severity_keywords = self._create_severity_mapping()
        
    def _create_team_name_mapping(self) -> Dict[str, str]:
        """Create mapping for team name variations."""
        return {
            # Full names to standard names
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa',
            'Bournemouth': 'Bournemouth',
            'AFC Bournemouth': 'Bournemouth',
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
            'Manchester City': 'Manchester City',
            'Man City': 'Manchester City',
            'City': 'Manchester City',
            'Manchester United': 'Manchester United',
            'Man Utd': 'Manchester United',
            'Man United': 'Manchester United',
            'United': 'Manchester United',
            'Newcastle': 'Newcastle',
            'Newcastle United': 'Newcastle',
            'Nottingham Forest': 'Nottingham Forest',
            "Nott'm Forest": 'Nottingham Forest',
            'Forest': 'Nottingham Forest',
            'Sheffield United': 'Sheffield United',
            'Sheffield Utd': 'Sheffield United',
            'Tottenham': 'Tottenham',
            'Tottenham Hotspur': 'Tottenham',
            'Spurs': 'Tottenham',
            'West Ham': 'West Ham',
            'West Ham United': 'West Ham',
            'Wolves': 'Wolves',
            'Wolverhampton': 'Wolves',
            'Wolverhampton Wanderers': 'Wolves'
        }
    
    def _create_severity_mapping(self) -> Dict[str, int]:
        """Create mapping for injury severity based on keywords."""
        return {
            # Severity 1: Minor issues
            'knock': 1,
            'minor': 1,
            'slight': 1,
            'fatigue': 1,
            'tight': 1,
            'precaution': 1,
            
            # Severity 2: Short-term injuries
            'strain': 2,
            'bruise': 2,
            'cut': 2,
            'illness': 2,
            
            # Severity 3: Medium-term injuries
            'muscle': 3,
            'hamstring': 3,
            'calf': 3,
            'groin': 3,
            'thigh': 3,
            
            # Severity 4: Serious injuries
            'ankle': 4,
            'knee': 4,
            'shoulder': 4,
            'back': 4,
            'hip': 4,
            'fracture': 4,
            
            # Severity 5: Long-term injuries
            'surgery': 5,
            'operation': 5,
            'torn': 5,
            'rupture': 5,
            'cruciate': 5,
            'achilles': 5
        }
    
    def create_empty_template(self, filepath: str = None) -> pd.DataFrame:
        """Create an empty injury data template."""
        template = pd.DataFrame({
            'player_name': [],
            'team': [],
            'injury_type': [],
            'status': [],
            'expected_return': [],
            'severity': [],
            'availability_probability': [],
            'last_updated': [],
            'source': [],
            'notes': []
        })
        
        if filepath:
            template.to_csv(filepath, index=False)
            print(f"✅ Empty template created: {filepath}")
        
        return template
    
    def process_manual_data(self, filepath: str) -> pd.DataFrame:
        """Process manually collected injury data."""
        try:
            df = pd.read_csv(filepath)
            print(f"📂 Loading manual injury data: {filepath}")
            print(f"📊 Found {len(df)} injury records")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return pd.DataFrame()
        
        if df.empty:
            print("⚠️ No data found in file")
            return df
        
        # Standardize team names
        df['team_standardized'] = df['team'].map(self.team_name_mapping).fillna(df['team'])
        
        # Calculate severity if not provided
        if 'severity' not in df.columns or df['severity'].isna().any():
            df['calculated_severity'] = df['injury_type'].str.lower().apply(
                self._calculate_severity_from_text
            )
            
            # Use provided severity if available, otherwise use calculated
            if 'severity' in df.columns:
                df['severity'] = df['severity'].fillna(df['calculated_severity'])
            else:
                df['severity'] = df['calculated_severity']
        
        # Calculate availability probability if not provided
        if 'availability_probability' not in df.columns or df['availability_probability'].isna().any():
            df['calculated_availability'] = df.apply(
                lambda row: self._calculate_availability(row['status'], row['severity']), axis=1
            )
            
            if 'availability_probability' in df.columns:
                df['availability_probability'] = df['availability_probability'].fillna(df['calculated_availability'])
            else:
                df['availability_probability'] = df['calculated_availability']
        
        # Add processing metadata
        df['processed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['data_source'] = 'manual_entry'
        
        # Validate data
        df = self._validate_injury_data(df)
        
        print(f"✅ Processed {len(df)} injury records")
        print(f"📊 Severity distribution: {df['severity'].value_counts().to_dict()}")
        print(f"📊 Average availability: {df['availability_probability'].mean():.2f}")
        
        return df
    
    def _calculate_severity_from_text(self, injury_text: str) -> int:
        """Calculate injury severity from text description."""
        if pd.isna(injury_text):
            return 2  # Default medium severity
        
        injury_text = str(injury_text).lower()
        
        # Find highest severity keyword
        max_severity = 2  # Default
        for keyword, severity in self.severity_keywords.items():
            if keyword in injury_text:
                max_severity = max(max_severity, severity)
        
        return max_severity
    
    def _calculate_availability(self, status: str, severity: int) -> float:
        """Calculate availability probability based on status and severity."""
        if pd.isna(status):
            status = 'unknown'
        
        status = str(status).lower()
        
        # Base probability from status
        status_probs = {
            'available': 0.95,
            'fit': 0.95,
            'training': 0.85,
            'back in training': 0.80,
            'doubt': 0.60,
            'major doubt': 0.30,
            'unlikely': 0.25,
            'out': 0.05,
            'ruled out': 0.02,
            'surgery': 0.01,
            'long-term': 0.01
        }
        
        base_prob = 0.50  # Default
        for status_key, prob in status_probs.items():
            if status_key in status:
                base_prob = prob
                break
        
        # Adjust based on severity (1=minor, 5=severe)
        severity_adjustment = (6 - severity) / 5  # Higher severity = lower availability
        
        final_prob = base_prob * severity_adjustment
        return max(0.01, min(0.99, final_prob))  # Clamp between 1% and 99%
    
    def _validate_injury_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean injury data."""
        initial_count = len(df)
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['player_name', 'team'])
        
        # Ensure severity is in valid range
        df['severity'] = df['severity'].clip(1, 5)
        
        # Ensure availability is in valid range
        df['availability_probability'] = df['availability_probability'].clip(0, 1)
        
        # Remove duplicate player entries (keep most recent)
        df = df.sort_values('last_updated').drop_duplicates(subset=['player_name', 'team'], keep='last')
        
        final_count = len(df)
        if final_count < initial_count:
            print(f"⚠️ Removed {initial_count - final_count} invalid/duplicate records")
        
        return df
    
    def merge_with_existing(self, new_data: pd.DataFrame, existing_file: str = None) -> pd.DataFrame:
        """Merge new injury data with existing data."""
        if existing_file and os.path.exists(existing_file):
            try:
                existing_df = pd.read_csv(existing_file)
                print(f"📂 Loading existing injury data: {len(existing_df)} records")
                
                # Combine and remove duplicates (favor newer data)
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                combined_df = combined_df.sort_values('last_updated').drop_duplicates(
                    subset=['player_name', 'team'], keep='last'
                )
                
                print(f"✅ Merged data: {len(combined_df)} total records")
                return combined_df
            except Exception as e:
                print(f"⚠️ Error loading existing data: {e}")
                return new_data
        else:
            return new_data
    
    def export_for_fpl(self, df: pd.DataFrame, output_file: str):
        """Export processed injury data in FPL-compatible format."""
        fpl_format = df[['player_name', 'team_standardized', 'injury_type', 'severity', 
                        'availability_probability', 'expected_return', 'last_updated']].copy()
        
        fpl_format.columns = ['player_name', 'team', 'injury_type', 'injury_severity',
                             'availability_probability', 'expected_return', 'last_updated']
        
        fpl_format.to_csv(output_file, index=False)
        print(f"✅ FPL-compatible injury data exported: {output_file}")
        
        return fpl_format


def main():
    """Example usage of the manual injury processor."""
    processor = ManualInjuryProcessor()
    
    # Create empty template
    template_file = "injury_data_template.csv"
    processor.create_empty_template(template_file)
    
    print("\n📋 Manual Injury Data Collection Instructions:")
    print("1. Fill out the template with injury data from Sky Sports or BBC Sport")
    print("2. Use the process_manual_data() function to clean and validate")
    print("3. Export in FPL-compatible format for integration")
    
    example_data = pd.DataFrame({
        'player_name': ['Mohamed Salah', 'Erling Haaland'],
        'team': ['Liverpool', 'Man City'],
        'injury_type': ['Hamstring strain', 'Ankle injury'],
        'status': ['Doubt', 'Out'],
        'expected_return': ['2024-08-20', '2024-08-25'],
        'last_updated': [datetime.now().strftime('%Y-%m-%d')] * 2
    })
    
    processed = processor.process_manual_data("example")
    print(f"Example processing completed")


if __name__ == "__main__":
    main()