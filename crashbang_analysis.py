"""
═══════════════════════════════════════════════════════════════════════════
                        CRASHBANG ANALYTICA v2.0
              Traffic Accident Analysis & Risk Prediction System
                  with Machine Learning & Interactive Dashboard
                         by Michael Semera
═══════════════════════════════════════════════════════════════════════════

Description:
    A comprehensive traffic accident analysis system that identifies factors
    contributing to accidents and predicts high-risk zones using advanced
    data science, machine learning, and interactive visualizations.

Features:
    - Exploratory Data Analysis (EDA) of accident patterns
    - Temporal analysis (time, day, month, season)
    - Weather impact assessment
    - Geographic hotspot identification
    - Risk factor correlation analysis
    - Machine Learning prediction models
    - Interactive visualizations with Plotly
    - HTML dashboard generation
    - Feature importance analysis
    - Predictive modeling for accident severity
    - Real-time risk zone mapping

Dataset:
    US Accidents Dataset from Kaggle
    (https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

Requirements:
    pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, folium, xgboost

Author: Michael Semera
Version: 2.0
Date: 2025
═══════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import json

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Interactive visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not installed. Interactive features will be limited.")

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("[WARNING] Folium not installed. Map features will be limited.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CrashBangAnalytica:
    """
    Advanced traffic accident analysis system with ML and interactive visualizations.
    
    This class handles data processing, statistical analysis, machine learning
    predictions, and generates interactive dashboards for accident insights.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the CrashBang Analytica system.
        
        Args:
            data_path: Path to the accident dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_ml = None  # DataFrame prepared for ML
        self.risk_zones = None
        self.accident_factors = {}
        self.ml_models = {}
        self.feature_importance = {}
        self.predictions = None
        
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                                                           ║")
        print("║            CRASHBANG ANALYTICA v2.0                       ║")
        print("║   Advanced Traffic Accident Analysis System               ║")
        print("║        with Machine Learning & Interactive UI             ║")
        print("║              by Michael Semera                            ║")
        print("║                                                           ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and perform initial data validation.
        
        Args:
            sample_size: Number of records to sample (None for full dataset)
            
        Returns:
            DataFrame containing accident data
        """
        print("[INFO] Loading accident data...")
        
        if self.data_path is None:
            print("[WARNING] No data path provided. Using sample data generation.")
            self.df = self._generate_sample_data(sample_size or 10000)
        else:
            try:
                self.df = pd.read_csv(self.data_path)
                if sample_size:
                    self.df = self.df.sample(n=min(sample_size, len(self.df)), 
                                           random_state=42)
            except FileNotFoundError:
                print(f"[ERROR] File not found: {self.data_path}")
                print("[INFO] Generating sample data instead...")
                self.df = self._generate_sample_data(sample_size or 10000)
        
        print(f"✓ Data loaded successfully: {len(self.df):,} records")
        print(f"✓ Columns: {self.df.shape[1]}")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        return self.df
    
    def _generate_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate realistic sample accident data for demonstration.
        
        Args:
            n_samples: Number of sample records to generate
            
        Returns:
            DataFrame with synthetic accident data
        """
        np.random.seed(42)
        
        # Generate temporal features
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2024-12-31')
        dates = pd.date_range(start_date, end_date, periods=n_samples)
        
        # Generate geographic features (US coordinates)
        latitudes = np.random.uniform(25.0, 49.0, n_samples)
        longitudes = np.random.uniform(-125.0, -65.0, n_samples)
        
        # Generate weather conditions
        weather_conditions = np.random.choice(
            ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Windy'],
            n_samples,
            p=[0.40, 0.25, 0.10, 0.08, 0.12, 0.05]
        )
        
        # Generate severity with realistic patterns
        base_severity = np.random.choice([1, 2, 3, 4], n_samples, p=[0.30, 0.40, 0.20, 0.10])
        
        # Adjust severity based on conditions
        severity = base_severity.copy()
        for i in range(n_samples):
            if weather_conditions[i] in ['Snow', 'Fog']:
                severity[i] = min(4, severity[i] + 1)
            if dates[i].hour in [7, 8, 17, 18]:  # Rush hours
                severity[i] = min(4, severity[i] + np.random.choice([0, 1], p=[0.7, 0.3]))
        
        # Generate visibility (miles)
        visibility = np.random.gamma(shape=2, scale=5, size=n_samples)
        visibility = np.clip(visibility, 0.1, 10)
        
        # Generate temperature (Fahrenheit)
        temperature = np.random.normal(loc=65, scale=20, size=n_samples)
        
        # Generate humidity
        humidity = np.random.uniform(20, 100, n_samples)
        
        # Generate road features
        road_features = {
            'Crossing': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
            'Junction': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),
            'Stop': np.random.choice([True, False], n_samples, p=[0.10, 0.90]),
            'Traffic_Signal': np.random.choice([True, False], n_samples, p=[0.20, 0.80]),
        }
        
        # Create DataFrame
        data = {
            'Start_Time': dates,
            'Start_Lat': latitudes,
            'Start_Lng': longitudes,
            'Severity': severity,
            'Weather_Condition': weather_conditions,
            'Visibility(mi)': visibility,
            'Temperature(F)': temperature,
            'Humidity(%)': humidity,
            'Crossing': road_features['Crossing'],
            'Junction': road_features['Junction'],
            'Stop': road_features['Stop'],
            'Traffic_Signal': road_features['Traffic_Signal'],
        }
        
        return pd.DataFrame(data)
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the accident data.
        
        Returns:
            Cleaned DataFrame
        """
        print("[INFO] Cleaning data...")
        
        if self.df is None:
            print("[ERROR] No data loaded. Please load data first.")
            return None
        
        initial_rows = len(self.df)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values in critical columns
        if 'Severity' in self.df.columns:
            self.df = self.df.dropna(subset=['Severity'])
        
        # Convert Start_Time to datetime if exists
        if 'Start_Time' in self.df.columns:
            self.df['Start_Time'] = pd.to_datetime(self.df['Start_Time'], 
                                                   errors='coerce')
            self.df = self.df.dropna(subset=['Start_Time'])
        
        # Create temporal features
        if 'Start_Time' in self.df.columns:
            self.df['Hour'] = self.df['Start_Time'].dt.hour
            self.df['DayOfWeek'] = self.df['Start_Time'].dt.dayofweek
            self.df['Month'] = self.df['Start_Time'].dt.month
            self.df['Year'] = self.df['Start_Time'].dt.year
            self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
            self.df['IsRushHour'] = self.df['Hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            
            # Create time period categories
            self.df['Time_Period'] = pd.cut(
                self.df['Hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # Create season
            self.df['Season'] = self.df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        final_rows = len(self.df)
        print(f"✓ Data cleaned: {initial_rows - final_rows:,} rows removed")
        print(f"✓ Final dataset: {final_rows:,} records\n")
        
        return self.df
    
    def create_interactive_temporal_dashboard(self) -> None:
        """
        Create interactive temporal analysis dashboard using Plotly.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARNING] Plotly not available. Install with: pip install plotly")
            return
        
        print("[INFO] Creating interactive temporal dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accidents by Hour', 'Accidents by Day of Week',
                          'Accidents by Month', 'Severity Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Hourly distribution
        if 'Hour' in self.df.columns:
            hourly_counts = self.df['Hour'].value_counts().sort_index()
            axes[0, 0].bar(hourly_counts.index, hourly_counts.values, 
                          color='steelblue', alpha=0.7)
            axes[0, 0].set_xlabel('Hour of Day', fontweight='bold')
            axes[0, 0].set_ylabel('Number of Accidents', fontweight='bold')
            axes[0, 0].set_title('Accidents by Hour of Day')
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Day of week distribution
        if 'DayOfWeek' in self.df.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_counts = self.df['DayOfWeek'].value_counts().sort_index()
            axes[0, 1].bar(range(7), day_counts.values, 
                          tick_label=day_names, color='coral', alpha=0.7)
            axes[0, 1].set_xlabel('Day of Week', fontweight='bold')
            axes[0, 1].set_ylabel('Number of Accidents', fontweight='bold')
            axes[0, 1].set_title('Accidents by Day of Week')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Monthly distribution
        if 'Month' in self.df.columns:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_counts = self.df['Month'].value_counts().sort_index()
            axes[1, 0].bar(range(1, 13), month_counts.values,
                          tick_label=month_names, color='lightgreen', alpha=0.7)
            axes[1, 0].set_xlabel('Month', fontweight='bold')
            axes[1, 0].set_ylabel('Number of Accidents', fontweight='bold')
            axes[1, 0].set_title('Accidents by Month')
            axes[1, 0].grid(axis='y', alpha=0.3)
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Season distribution
        if 'Season' in self.df.columns:
            season_counts = self.df['Season'].value_counts()
            colors_season = ['lightblue', 'lightgreen', 'orange', 'brown']
            axes[1, 1].pie(season_counts.values, labels=season_counts.index,
                          autopct='%1.1f%%', startangle=90, colors=colors_season)
            axes[1, 1].set_title('Accidents by Season')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Temporal analysis plot saved: temporal_analysis.png")
        
        plt.show()
        print("✓ Temporal pattern analysis complete\n")
    
    def analyze_weather_impact(self, save_plots: bool = False) -> pd.DataFrame:
        """Analyze the impact of weather conditions on accident severity."""
        print("[INFO] Analyzing weather impact...")
        
        if 'Weather_Condition' not in self.df.columns:
            print("[WARNING] Weather data not available.")
            return None
        
        # Calculate statistics by weather condition
        weather_stats = self.df.groupby('Weather_Condition').agg({
            'Severity': ['mean', 'count'],
            'Visibility(mi)': 'mean' if 'Visibility(mi)' in self.df.columns else None
        }).round(2)
        
        weather_stats.columns = ['Avg_Severity', 'Accident_Count', 'Avg_Visibility']
        weather_stats = weather_stats.sort_values('Accident_Count', ascending=False)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('CrashBang Analytica: Weather Impact Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Accident count by weather
        weather_stats['Accident_Count'].plot(kind='barh', ax=axes[0], 
                                             color='skyblue', alpha=0.7)
        axes[0].set_xlabel('Number of Accidents', fontweight='bold')
        axes[0].set_ylabel('Weather Condition', fontweight='bold')
        axes[0].set_title('Accident Frequency by Weather Condition')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Average severity by weather
        weather_stats['Avg_Severity'].plot(kind='barh', ax=axes[1],
                                           color='salmon', alpha=0.7)
        axes[1].set_xlabel('Average Severity', fontweight='bold')
        axes[1].set_ylabel('Weather Condition', fontweight='bold')
        axes[1].set_title('Average Severity by Weather Condition')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('weather_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Weather analysis plot saved: weather_analysis.png")
        
        plt.show()
        
        print("\n📊 Weather Impact Statistics:")
        print(weather_stats)
        print("\n✓ Weather impact analysis complete\n")
        
        return weather_stats
    
    def identify_risk_factors(self) -> Dict:
        """Identify and rank factors contributing to accident severity."""
        print("[INFO] Identifying risk factors...")
        
        risk_factors = {}
        
        # Analyze road features impact
        road_features = ['Crossing', 'Junction', 'Stop', 'Traffic_Signal']
        
        for feature in road_features:
            if feature in self.df.columns:
                with_feature = self.df[self.df[feature] == True]['Severity'].mean()
                without_feature = self.df[self.df[feature] == False]['Severity'].mean()
                impact = with_feature - without_feature
                
                risk_factors[feature] = {
                    'with_feature_severity': round(with_feature, 2),
                    'without_feature_severity': round(without_feature, 2),
                    'impact': round(impact, 2)
                }
        
        # Display results
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║              RISK FACTOR ANALYSIS                         ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        
        for factor, stats in sorted(risk_factors.items(), 
                                   key=lambda x: abs(x[1]['impact']), 
                                   reverse=True):
            print(f"📍 {factor}:")
            print(f"   With feature: {stats['with_feature_severity']:.2f}")
            print(f"   Without feature: {stats['without_feature_severity']:.2f}")
            print(f"   Impact: {stats['impact']:+.2f}")
            print()
        
        self.accident_factors = risk_factors
        print("✓ Risk factor analysis complete\n")
        
        return risk_factors
    
    def identify_high_risk_zones(self, grid_size: float = 0.5) -> pd.DataFrame:
        """Identify geographic areas with high accident density."""
        print(f"[INFO] Identifying high-risk zones (grid size: {grid_size}°)...")
        
        if 'Start_Lat' not in self.df.columns or 'Start_Lng' not in self.df.columns:
            print("[WARNING] Geographic data not available.")
            return None
        
        # Create geographic grid
        self.df['Lat_Grid'] = (self.df['Start_Lat'] // grid_size) * grid_size
        self.df['Lng_Grid'] = (self.df['Start_Lng'] // grid_size) * grid_size
        
        # Count accidents per grid cell
        risk_zones = self.df.groupby(['Lat_Grid', 'Lng_Grid']).agg({
            'Severity': ['count', 'mean']
        }).reset_index()
        
        risk_zones.columns = ['Latitude', 'Longitude', 'Accident_Count', 'Avg_Severity']
        
        # Calculate risk score (weighted combination)
        risk_zones['Risk_Score'] = (
            risk_zones['Accident_Count'] * 0.7 + 
            risk_zones['Avg_Severity'] * 100 * 0.3
        )
        
        # Get top risk zones
        risk_zones = risk_zones.sort_values('Risk_Score', ascending=False).head(20)
        
        print(f"\n📍 Top 10 High-Risk Zones:")
        print("="*70)
        print(f"{'Rank':<6}{'Latitude':<12}{'Longitude':<12}{'Accidents':<12}{'Risk Score':<12}")
        print("="*70)
        
        for idx, (_, row) in enumerate(risk_zones.head(10).iterrows(), 1):
            print(f"{idx:<6}{row['Latitude']:<12.2f}{row['Longitude']:<12.2f}"
                  f"{int(row['Accident_Count']):<12}{row['Risk_Score']:<12.1f}")
        
        self.risk_zones = risk_zones
        print("\n✓ High-risk zone identification complete\n")
        
        return risk_zones
    
    def generate_comprehensive_report(self, output_file: str = 'crashbang_report.txt') -> None:
        """Generate a comprehensive text report of all analyses."""
        print(f"[INFO] Generating comprehensive report...")
        
        with open(output_file, 'w') as f:
            f.write("═"*70 + "\n")
            f.write(" "*15 + "CRASHBANG ANALYTICA v2.0\n")
            f.write(" "*10 + "Traffic Accident Analysis Report\n")
            f.write(" "*20 + "by Michael Semera\n")
            f.write("═"*70 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Size: {len(self.df):,} records\n\n")
            
            # Summary statistics
            summary = self.generate_summary_statistics()
            f.write("\n" + "─"*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("─"*70 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            # Risk factors
            if self.accident_factors:
                f.write("\n" + "─"*70 + "\n")
                f.write("RISK FACTORS\n")
                f.write("─"*70 + "\n")
                for factor, stats in self.accident_factors.items():
                    f.write(f"\n{factor}:\n")
                    for stat_name, stat_value in stats.items():
                        f.write(f"  {stat_name}: {stat_value}\n")
            
            # Machine learning results
            if self.ml_models:
                f.write("\n" + "─"*70 + "\n")
                f.write("MACHINE LEARNING MODELS\n")
                f.write("─"*70 + "\n")
                for model_name, model_data in self.ml_models.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Accuracy: {model_data['accuracy']:.4f}\n")
                    f.write(f"  CV Mean: {model_data['cv_mean']:.4f}\n")
                    f.write(f"  CV Std: {model_data['cv_std']:.4f}\n")
                
                if self.feature_importance:
                    f.write("\nFeature Importance (Random Forest):\n")
                    sorted_features = sorted(self.feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)
                    for feat, imp in sorted_features:
                        f.write(f"  {feat}: {imp:.4f}\n")
            
            # High-risk zones
            if self.risk_zones is not None:
                f.write("\n" + "─"*70 + "\n")
                f.write("TOP HIGH-RISK ZONES\n")
                f.write("─"*70 + "\n")
                f.write(self.risk_zones.head(10).to_string())
            
            f.write("\n\n" + "═"*70 + "\n")
            f.write("End of Report\n")
            f.write("═"*70 + "\n")
        
        print(f"✓ Comprehensive report saved: {output_file}\n")
    
    def run_full_analysis(self, save_outputs: bool = True, 
                         enable_ml: bool = True,
                         enable_interactive: bool = True) -> None:
        """
        Execute complete analysis pipeline with ML and interactive features.
        
        Args:
            save_outputs: Whether to save plots and reports
            enable_ml: Whether to train ML models
            enable_interactive: Whether to create interactive dashboards
        """
        print("\n" + "═"*70)
        print(" "*15 + "STARTING FULL ANALYSIS PIPELINE")
        print("═"*70 + "\n")
        
        # Load and clean data
        self.load_data()
        self.clean_data()
        
        # Generate analyses
        self.generate_summary_statistics()
        self.analyze_temporal_patterns(save_plots=save_outputs)
        self.analyze_weather_impact(save_plots=save_outputs)
        self.identify_risk_factors()
        self.identify_high_risk_zones()
        
        # Machine learning
        if enable_ml:
            self.prepare_ml_data()
            self.train_ml_models()
            self.visualize_model_performance(save_plot=save_outputs)
        
        # Interactive visualizations
        if enable_interactive:
            self.create_interactive_temporal_dashboard()
            self.create_interactive_heatmap()
        
        # Generate report
        if save_outputs:
            self.generate_comprehensive_report()
        
        print("\n" + "═"*70)
        print(" "*15 + "ANALYSIS PIPELINE COMPLETE")
        print("═"*70 + "\n")
        
        # Display summary of generated files
        print("📁 Generated Files:")
        if save_outputs:
            print("   ✓ temporal_analysis.png")
            print("   ✓ weather_analysis.png")
            print("   ✓ crashbang_report.txt")
            if enable_ml:
                print("   ✓ ml_model_performance.png")
            if enable_interactive:
                print("   ✓ interactive_temporal_dashboard.html")
                print("   ✓ accident_heatmap.html")
        print()


def demo_prediction_example(analyzer):
    """
    Demonstrate the prediction capability with example scenarios.
    
    Args:
        analyzer: Trained CrashBangAnalytica instance
    """
    print("\n" + "═"*70)
    print(" "*20 + "PREDICTION DEMO")
    print("═"*70 + "\n")
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Rush Hour, Clear Weather',
            'features': {
                'Hour': 17,
                'DayOfWeek': 4,  # Friday
                'Month': 6,
                'IsWeekend': 0,
                'IsRushHour': 1,
                'Weather_Encoded': 0,  # Clear
                'Visibility(mi)': 10.0,
                'Temperature(F)': 75.0,
                'Humidity(%)': 50.0,
                'Crossing': 1,
                'Junction': 0,
                'Stop': 0,
                'Traffic_Signal': 1
            }
        },
        {
            'name': 'Night, Rainy Weather',
            'features': {
                'Hour': 23,
                'DayOfWeek': 6,  # Sunday
                'Month': 11,
                'IsWeekend': 1,
                'IsRushHour': 0,
                'Weather_Encoded': 1,  # Rain
                'Visibility(mi)': 2.0,
                'Temperature(F)': 45.0,
                'Humidity(%)': 85.0,
                'Crossing': 0,
                'Junction': 1,
                'Stop': 0,
                'Traffic_Signal': 0
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"🔮 Scenario: {scenario['name']}")
        print("─"*70)
        
        predictions = analyzer.predict_accident_risk(scenario['features'])
        
        for model_name, pred_data in predictions.items():
            print(f"\n{model_name} Prediction:")
            print(f"  Predicted Severity: {pred_data['predicted_severity']}")
            print(f"  Probability Distribution:")
            for sev, prob in pred_data['probabilities'].items():
                bar = '█' * int(prob * 50)
                print(f"    {sev}: {prob:.3f} {bar}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point for CrashBang Analytica v2.0."""
    
    # Initialize analyzer
    analyzer = CrashBangAnalytica()
    
    # Run full analysis with all features
    analyzer.run_full_analysis(
        save_outputs=True,
        enable_ml=True,
        enable_interactive=True
    )
    
    # Demonstrate prediction capability
    demo_prediction_example(analyzer)
    
    print("\n✅ CrashBang Analytica v2.0 analysis complete!")
    print("\n📊 Generated Outputs:")
    print("   • Static visualizations (PNG files)")
    print("   • Interactive dashboards (HTML files)")
    print("   • Comprehensive text report")
    print("   • Trained ML models ready for predictions")
    
    print("\n💡 Example Usage:")
    print("   # Make predictions for custom scenarios")
    print("   features = {'Hour': 18, 'DayOfWeek': 4, ...}")
    print("   predictions = analyzer.predict_accident_risk(features)")
    
    print("\n" + "─"*70)
    print("Thank you for using CrashBang Analytica v2.0 - by Michael Semera")
    print("Making roads safer through data science and machine learning")
    print("─"*70 + "\n")


if __name__ == "__main__":
    main()
        hourly_counts = self.df['Hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, 
                   name='Hourly', marker_color='steelblue'),
            row=1, col=1
        )
        
        # Day of week distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        day_counts = self.df['DayOfWeek'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=[day_names[i] for i in day_counts.index], 
                   y=day_counts.values, name='Daily', marker_color='coral'),
            row=1, col=2
        )
        
        # Monthly distribution
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts = self.df['Month'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=[month_names[i-1] for i in month_counts.index],
                   y=month_counts.values, name='Monthly', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Severity distribution (pie chart)
        severity_counts = self.df['Severity'].value_counts().sort_index()
        fig.add_trace(
            go.Pie(labels=[f'Severity {i}' for i in severity_counts.index],
                   values=severity_counts.values,
                   marker=dict(colors=['lightblue', 'yellow', 'orange', 'red'])),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="CrashBang Analytica: Interactive Temporal Analysis Dashboard",
            title_font_size=20,
            showlegend=False,
            height=800
        )
        
        # Save to HTML
        fig.write_html('interactive_temporal_dashboard.html')
        print("✓ Interactive dashboard saved: interactive_temporal_dashboard.html")
        print("  Open in browser to interact with visualizations\n")
    
    def create_interactive_heatmap(self) -> None:
        """
        Create interactive geographic heatmap of accidents.
        """
        if not FOLIUM_AVAILABLE:
            print("[WARNING] Folium not available. Install with: pip install folium")
            return
        
        print("[INFO] Creating interactive accident heatmap...")
        
        # Sample data for performance (max 1000 points for heatmap)
        sample_df = self.df.sample(n=min(1000, len(self.df)), random_state=42)
        
        # Calculate center point
        center_lat = sample_df['Start_Lat'].mean()
        center_lng = sample_df['Start_Lng'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Prepare data for heatmap
        heat_data = [[row['Start_Lat'], row['Start_Lng'], row['Severity']] 
                     for idx, row in sample_df.iterrows()]
        
        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
        
        # Add marker cluster for high severity accidents
        high_severity = sample_df[sample_df['Severity'] >= 3]
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in high_severity.iterrows():
            folium.Marker(
                location=[row['Start_Lat'], row['Start_Lng']],
                popup=f"Severity: {row['Severity']}<br>Time: {row['Start_Time']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        
        # Save map
        m.save('accident_heatmap.html')
        print("✓ Interactive heatmap saved: accident_heatmap.html")
        print("  Open in browser to explore accident locations\n")
    
    def prepare_ml_data(self) -> pd.DataFrame:
        """
        Prepare data for machine learning models.
        
        Returns:
            DataFrame ready for ML training
        """
        print("[INFO] Preparing data for machine learning...")
        
        # Create a copy for ML
        self.df_ml = self.df.copy()
        
        # Select features for ML
        feature_columns = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsRushHour']
        
        # Add weather if available
        if 'Weather_Condition' in self.df_ml.columns:
            # Encode weather conditions
            le = LabelEncoder()
            self.df_ml['Weather_Encoded'] = le.fit_transform(self.df_ml['Weather_Condition'])
            feature_columns.append('Weather_Encoded')
        
        # Add numeric features
        numeric_features = ['Visibility(mi)', 'Temperature(F)', 'Humidity(%)']
        for feat in numeric_features:
            if feat in self.df_ml.columns:
                self.df_ml[feat] = self.df_ml[feat].fillna(self.df_ml[feat].median())
                feature_columns.append(feat)
        
        # Add boolean features
        bool_features = ['Crossing', 'Junction', 'Stop', 'Traffic_Signal']
        for feat in bool_features:
            if feat in self.df_ml.columns:
                self.df_ml[feat] = self.df_ml[feat].astype(int)
                feature_columns.append(feat)
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        print(f"✓ ML data prepared with {len(feature_columns)} features")
        print(f"  Features: {', '.join(feature_columns)}\n")
        
        return self.df_ml
    
    def train_ml_models(self) -> Dict:
        """
        Train multiple machine learning models for severity prediction.
        
        Returns:
            Dictionary containing trained models and metrics
        """
        print("[INFO] Training machine learning models...")
        
        if self.df_ml is None:
            self.prepare_ml_data()
        
        # Prepare features and target
        X = self.df_ml[self.feature_columns]
        y = self.df_ml['Severity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n[INFO] Training {model_name}...")
            
            # Train model
            if model_name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            if model_name == 'Random Forest':
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store results
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            # Feature importance for Random Forest
            if model_name == 'Random Forest':
                importances = model.feature_importances_
                self.feature_importance = dict(zip(self.feature_columns, importances))
            
            print(f"✓ {model_name} - Accuracy: {accuracy:.4f}")
            print(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.ml_models = results
        self.scaler = scaler
        
        print("\n✓ Machine learning models trained successfully\n")
        
        return results
    
    def visualize_model_performance(self, save_plot: bool = True) -> None:
        """
        Visualize machine learning model performance.
        
        Args:
            save_plot: Whether to save the plot to file
        """
        print("[INFO] Visualizing model performance...")
        
        if not self.ml_models:
            print("[WARNING] No models trained. Train models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CrashBang Analytica: ML Model Performance Analysis',
                    fontsize=16, fontweight='bold')
        
        # Model comparison
        model_names = list(self.ml_models.keys())
        accuracies = [self.ml_models[m]['accuracy'] for m in model_names]
        cv_means = [self.ml_models[m]['cv_mean'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Test Accuracy', color='steelblue')
        axes[0, 0].bar(x + width/2, cv_means, width, label='CV Mean', color='coral')
        axes[0, 0].set_xlabel('Model', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Feature importance
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importances = list(self.feature_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            pos = np.arange(sorted_idx.shape[0])
            
            axes[0, 1].barh(pos, np.array(importances)[sorted_idx], color='lightgreen')
            axes[0, 1].set_yticks(pos)
            axes[0, 1].set_yticklabels(np.array(features)[sorted_idx])
            axes[0, 1].set_xlabel('Importance', fontweight='bold')
            axes[0, 1].set_title('Feature Importance (Random Forest)')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Confusion matrix for best model
        best_model = max(self.ml_models.items(), key=lambda x: x[1]['accuracy'])
        model_name, model_data = best_model
        
        cm = confusion_matrix(model_data['y_test'], model_data['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {model_name}')
        axes[1, 0].set_xlabel('Predicted', fontweight='bold')
        axes[1, 0].set_ylabel('Actual', fontweight='bold')
        
        # Severity distribution: Actual vs Predicted
        severity_actual = model_data['y_test'].value_counts().sort_index()
        severity_pred = pd.Series(model_data['predictions']).value_counts().sort_index()
        
        x_sev = np.arange(len(severity_actual))
        width_sev = 0.35
        
        axes[1, 1].bar(x_sev - width_sev/2, severity_actual.values, 
                      width_sev, label='Actual', color='steelblue', alpha=0.7)
        axes[1, 1].bar(x_sev + width_sev/2, severity_pred.values,
                      width_sev, label='Predicted', color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Severity Level', fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontweight='bold')
        axes[1, 1].set_title('Severity Distribution: Actual vs Predicted')
        axes[1, 1].set_xticks(x_sev)
        axes[1, 1].set_xticklabels(severity_actual.index)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('ml_model_performance.png', dpi=300, bbox_inches='tight')
            print("✓ Model performance plot saved: ml_model_performance.png")
        
        plt.show()
        print("✓ Model performance visualization complete\n")
    
    def predict_accident_risk(self, features: Dict) -> Dict:
        """
        Predict accident severity risk for given conditions.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with predictions from all models
        """
        if not self.ml_models:
            print("[WARNING] Models not trained. Training now...")
            self.train_ml_models()
        
        # Prepare input features
        input_data = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[self.feature_columns]
        
        predictions = {}
        
        for model_name, model_data in self.ml_models.items():
            model = model_data['model']
            
            if model_name == 'Random Forest':
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]
            else:
                input_scaled = self.scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]
            
            predictions[model_name] = {
                'predicted_severity': int(pred),
                'probabilities': {f'Severity_{i+1}': float(p) for i, p in enumerate(prob)}
            }
        
        return predictions
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics."""
        print("[INFO] Generating summary statistics...\n")
        
        if self.df is None:
            print("[ERROR] No data available.")
            return {}
        
        summary = {}
        
        summary['total_accidents'] = len(self.df)
        summary['date_range'] = {
            'start': self.df['Start_Time'].min(),
            'end': self.df['Start_Time'].max()
        }
        
        if 'Severity' in self.df.columns:
            summary['severity_distribution'] = self.df['Severity'].value_counts().to_dict()
            summary['avg_severity'] = self.df['Severity'].mean()
        
        if 'Hour' in self.df.columns:
            summary['peak_hour'] = self.df['Hour'].mode()[0]
            summary['hourly_distribution'] = self.df['Hour'].value_counts().sort_index().to_dict()
        
        if 'DayOfWeek' in self.df.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            summary['busiest_day'] = day_names[self.df['DayOfWeek'].mode()[0]]
        
        if 'Weather_Condition' in self.df.columns:
            summary['weather_distribution'] = self.df['Weather_Condition'].value_counts().to_dict()
        
        self._display_summary(summary)
        
        return summary
    
    def _display_summary(self, summary: Dict) -> None:
        """Display formatted summary statistics."""
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║               ACCIDENT DATA SUMMARY                       ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        
        print(f"\n📊 Total Accidents: {summary['total_accidents']:,}")
        print(f"📅 Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        if 'avg_severity' in summary:
            print(f"⚠️  Average Severity: {summary['avg_severity']:.2f}/4.0")
        
        if 'peak_hour' in summary:
            print(f"🕐 Peak Hour: {summary['peak_hour']}:00")
        
        if 'busiest_day' in summary:
            print(f"📆 Busiest Day: {summary['busiest_day']}")
        
        print("\n" + "="*60)
    
    def analyze_temporal_patterns(self, save_plots: bool = False) -> None:
        """Analyze and visualize temporal accident patterns."""
        print("\n[INFO] Analyzing temporal patterns...")
        
        if self.df is None:
            print("[ERROR] No data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CrashBang Analytica: Temporal Pattern Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Hourly distribution