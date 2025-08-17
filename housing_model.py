import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import database_accessor

class SouthJordanRandomForestModel:
    def __init__(self):
        """Initialize the Random Forest model with optimized parameters"""
        self.model = RandomForestRegressor(
            n_estimators=200,           # More trees for better performance
            max_depth=15,               # Control overfitting
            min_samples_split=5,        # Prevent overfitting
            min_samples_leaf=2,         # Prevent overfitting
            max_features='sqrt',        # Feature sampling for diversity
            random_state=42,
            n_jobs=-1                   # Use all cores for faster training
        )
        
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_importances = None
        self.is_trained = False

        
    def load_and_preprocess_data(self):
        """Load and preprocess the housing data"""
        data_getter = database_accessor.DatabaseAccessor()
        df = data_getter.get_all_elements()

        print("Original data shape:", df.shape)
        print("\nOriginal columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Clean and convert data types
        # Convert price from string to numeric (remove $ and commas)
        if df['price'].dtype == 'object':
            df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Convert beds, baths, sqft to numeric
        df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
        df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
        df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
        
        # Handle days_on_market - convert "Listed" to 0, others to numeric
        df['days_on_market'] = df['days_on_market'].replace('Listed', 0)
        df['days_on_market'] = pd.to_numeric(df['days_on_market'], errors='coerce')
        df['days_on_market'] = df['days_on_market'].fillna(0)
        
        # Feature Engineering
        print("\nCreating engineered features...")
        
        # Basic ratio features
        df['price_per_sqft'] = df['price'] / df['sqft']
        df['bed_bath_ratio'] = df['beds'] / df['baths'].replace(0, 0.5)  # Avoid division by zero
        df['total_rooms'] = df['beds'] + df['baths']
        df['sqft_per_room'] = df['sqft'] / df['total_rooms'].replace(0, 1)
        
        # Market timing features
        df['is_new_listing'] = (df['days_on_market'] <= 1).astype(int)
        df['days_on_market_log'] = np.log1p(df['days_on_market'])
        
        # Property size categories
        df['size_category'] = pd.cut(df['sqft'], 
                                    bins=[0, 1500, 2500, 3500, float('inf')], 
                                    labels=['Small', 'Medium', 'Large', 'XLarge'])
        
        # Luxury indicator (top 25% of prices)
        price_75th = df['price'].quantile(0.75)
        df['is_luxury'] = (df['price'] >= price_75th).astype(int)
        
        # Interaction features (Random Forest can capture these, but explicit features help)
        df['bed_sqft_interaction'] = df['beds'] * df['sqft']
        df['bath_sqft_interaction'] = df['baths'] * df['sqft']
        
        # Encode categorical variables
        df['status_encoded'] = self.label_encoder.fit_transform(df['status'])
        df['size_category_encoded'] = pd.Categorical(df['size_category']).codes
        
        # Remove rows with missing critical data
        original_shape = df.shape[0]
        df = df.dropna(subset=['price', 'beds', 'baths', 'sqft'])
        
        print(f"\nData cleaning complete:")
        print(f"  Removed {original_shape - df.shape[0]} rows with missing critical data")
        print(f"  Final data shape: {df.shape}")
        print(f"  Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"  Average price: ${df['price'].mean():,.0f}")
        print(f"  Median price: ${df['price'].median():,.0f}")
        
        return df

    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nPrice Statistics:")
        print(df['price'].describe())
        
        print(f"\nProperty Characteristics:")
        print(f"  Beds range: {df['beds'].min():.0f} - {df['beds'].max():.0f}")
        print(f"  Baths range: {df['baths'].min():.0f} - {df['baths'].max():.0f}")
        print(f"  Sqft range: {df['sqft'].min():,.0f} - {df['sqft'].max():,.0f}")
        print(f"  Days on market: {df['days_on_market'].min():.0f} - {df['days_on_market'].max():.0f}")
        
        # Status distribution
        print(f"\nProperty Status Distribution:")
        print(df['status'].value_counts())
        
        # Size category distribution
        print(f"\nSize Category Distribution:")
        print(df['size_category'].value_counts())
        
        # Correlation analysis
        numeric_cols = [
            'price', 'beds', 'baths', 'sqft', 'days_on_market', 
            'price_per_sqft', 'bed_bath_ratio', 'total_rooms',
            'sqft_per_room', 'days_on_market_log', 'bed_sqft_interaction'
        ]
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        correlation_matrix = df[available_cols].corr()
        
        print(f"\nTop correlations with price:")
        price_corr = correlation_matrix['price'].abs().sort_values(ascending=False)
        for feature, corr in price_corr.items():
            if feature != 'price':
                print(f"  {feature}: {corr:.3f}")
        
        return correlation_matrix

    def prepare_features(self, df):
        """Prepare features for Random Forest modeling"""
        # Select features for the model
        feature_columns = [
            # Basic property features
            'beds', 'baths', 'sqft',
            # Market features  
            'days_on_market', 'status_encoded',
            # Engineered features
            'bed_bath_ratio', 'total_rooms', 'sqft_per_room',
            'is_new_listing', 'days_on_market_log',
            'size_category_encoded', 'is_luxury',
            'bed_sqft_interaction', 'bath_sqft_interaction'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df['price'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        self.feature_names = available_features
        
        print(f"\nFeatures selected for Random Forest: {len(available_features)}")
        for i, feature in enumerate(available_features):
            print(f"  {i+1:2d}. {feature}")
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("\nPerforming hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [150, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use fewer parameter combinations if dataset is small
        if len(X_train) < 100:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt']
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=min(5, len(X_train) // 20 + 2),  # Adaptive CV folds
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:,.0f}")
        
        return grid_search.best_estimator_

    def train_model(self, X, y, tune_hyperparameters=True):
        """Train the Random Forest model"""
        print("\n" + "="*60)
        print("RANDOM FOREST MODEL TRAINING")
        print("="*60)
        
        # Stratified split based on price ranges to ensure balanced training/test sets
        price_bins = pd.qcut(y, q=min(5, len(y)//10), labels=False, duplicates='drop')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=price_bins if len(np.unique(price_bins)) > 1 else None
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Hyperparameter tuning
        if tune_hyperparameters and len(X_train) >= 50:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            print("Using default parameters (dataset too small for tuning)")
        
        # Train the model
        print("\nTraining Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Cross-validation
        cv_folds = min(5, len(X_train) // 10 + 2)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        cv_mse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store feature importances
        self.feature_importances = self.model.feature_importances_
        self.is_trained = True
        
        # Print results
        print(f"\n🎯 MODEL PERFORMANCE RESULTS:")
        print(f"{'='*40}")
        print(f"Train MSE:     ${train_mse:,.0f}")
        print(f"Test MSE:      ${test_mse:,.0f}")
        print(f"Train R²:      {train_r2:.3f}")
        print(f"Test R²:       {test_r2:.3f}")
        print(f"Test MAE:      ${test_mae:,.0f}")
        print(f"CV MSE:        ${cv_mse:,.0f} (±${cv_std*2:,.0f})")
        
        # Calculate percentage errors
        test_mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        print(f"Test MAPE:     {test_mape:.1f}%")
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'cv_mse': cv_mse,
            'cv_std': cv_std,
            'test_predictions': test_pred,
            'test_actual': y_test,
            'X_test': X_test
        }
        
        return results

    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.3f}")
        
        # Feature importance visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot top 15 features
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance - Top 15 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return importance_df

    def make_prediction(self, beds, baths, sqft, days_on_market=0, status='ACTIVE'):
        """Make a price prediction for a new house"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Create feature vector with all engineered features
        bed_bath_ratio = beds / baths if baths > 0 else beds
        total_rooms = beds + baths
        sqft_per_room = sqft / total_rooms if total_rooms > 0 else sqft
        is_new_listing = 1 if days_on_market <= 1 else 0
        days_on_market_log = np.log1p(days_on_market)
        
        # Size category encoding
        if sqft <= 1500:
            size_category_encoded = 0  # Small
        elif sqft <= 2500:
            size_category_encoded = 1  # Medium
        elif sqft <= 3500:
            size_category_encoded = 2  # Large
        else:
            size_category_encoded = 3  # XLarge
        
        # Status encoding (0 for ACTIVE, 1 for others)
        status_encoded = 0 if status == 'ACTIVE' else 1
        
        # Interaction features
        bed_sqft_interaction = beds * sqft
        bath_sqft_interaction = baths * sqft
        
        # Create feature dictionary
        feature_dict = {
            'beds': beds,
            'baths': baths,
            'sqft': sqft,
            'days_on_market': days_on_market,
            'status_encoded': status_encoded,
            'bed_bath_ratio': bed_bath_ratio,
            'total_rooms': total_rooms,
            'sqft_per_room': sqft_per_room,
            'is_new_listing': is_new_listing,
            'days_on_market_log': days_on_market_log,
            'size_category_encoded': size_category_encoded,
            'is_luxury': 0,  # Will be determined after prediction
            'bed_sqft_interaction': bed_sqft_interaction,
            'bath_sqft_interaction': bath_sqft_interaction
        }
        
        # Create feature vector in the same order as training
        features = np.array([[feature_dict.get(fname, 0) for fname in self.feature_names]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get prediction confidence (using standard deviation of tree predictions)
        tree_predictions = [tree.predict(features)[0] for tree in self.model.estimators_]
        prediction_std = np.std(tree_predictions)
        
        return {
            'predicted_price': prediction,
            'confidence_interval_95': (prediction - 1.96 * prediction_std, 
                                        prediction + 1.96 * prediction_std),
            'prediction_std': prediction_std
        }

    def analyze_prediction_errors(self, results):
        """Analyze prediction errors to identify model weaknesses"""
        print("\n" + "="*60)
        print("PREDICTION ERROR ANALYSIS")
        print("="*60)
        
        predictions = results['test_predictions']
        actuals = results['test_actual']
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        percent_errors = abs_errors / actuals * 100
        
        print(f"Error Statistics:")
        print(f"  Mean Error (bias):           ${np.mean(errors):,.0f}")
        print(f"  Median Absolute Error:       ${np.median(abs_errors):,.0f}")
        print(f"  95th Percentile Error:       ${np.percentile(abs_errors, 95):,.0f}")
        print(f"  Max Error:                   ${np.max(abs_errors):,.0f}")
        print(f"  Mean Absolute Percentage Error: {np.mean(percent_errors):.1f}%")
        
        # Identify problematic price ranges
        price_ranges = pd.cut(actuals, bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        
        print(f"\nError by Price Range:")
        for price_range in price_ranges.cat.categories:
            mask = price_ranges == price_range
            if mask.sum() > 0:
                range_mape = np.mean(percent_errors[mask])
                range_count = mask.sum()
                print(f"  {price_range:8s}: {range_mape:5.1f}% MAPE ({range_count:2d} properties)")
        
        return {
            'mean_error': np.mean(errors),
            'median_abs_error': np.median(abs_errors),
            'max_error': np.max(abs_errors),
            'mape': np.mean(percent_errors)
        }

    def run_full_analysis(self, tune_hyperparameters=True):
        """Run the complete Random Forest analysis pipeline"""
        print("🌲 SOUTH JORDAN RANDOM FOREST HOUSING PRICE MODEL")
        print("=" * 80)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Explore data
        #correlation_matrix = self.explore_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df)

        # Train model
        results = self.train_model(X, y, tune_hyperparameters)
        
        # Analyze feature importance
        #importance_df = self.analyze_feature_importance()
        
        # Analyze errors
        error_stats = self.analyze_prediction_errors(results)
        
        # Example predictions
        print("\n" + "="*60)
        print("EXAMPLE PREDICTIONS")
        print("="*60)
        
        examples = [
            {"beds": 3, "baths": 2, "sqft": 1800, "days_on_market": 5},
            {"beds": 4, "baths": 2.5, "sqft": 2400, "days_on_market": 0},
            {"beds": 4, "baths": 3, "sqft": 2800, "days_on_market": 15},
            {"beds": 5, "baths": 4, "sqft": 3500, "days_on_market": 2}
        ]
        
        for example in examples:
            pred_result = self.make_prediction(**example)
            pred_price = pred_result['predicted_price']
            ci_low, ci_high = pred_result['confidence_interval_95']
            
            print(f"  {example['beds']} bed, {example['baths']} bath, {example['sqft']:,} sqft")
            print(f"    → Predicted: ${pred_price:,.0f}")
            print(f"    → 95% CI: ${ci_low:,.0f} - ${ci_high:,.0f}")
            print()
        
        return {
            'model': self,
            'results': results,
            'error_stats': error_stats
        }

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    
    # Initialize and run the model
    model = SouthJordanRandomForestModel()
    results = model.run_full_analysis(tune_hyperparameters=False)
    
    print(f"\n✅ Random Forest model training complete!")
    print(f"🎯 Test R² Score: {results['results']['test_r2']:.3f}")
    print(f"📊 Ready to make predictions for South Jordan, UT homes")
    
    # Example prediction
    print(f"\n🏠 Example prediction:")
    example_pred = model.make_prediction(beds=4, baths=3, sqft=2500, days_on_market=7)
    print(f"   4 bed, 3 bath, 2,500 sqft home → ${example_pred['predicted_price']:,.0f}")