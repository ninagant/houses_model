# South Jordan Housing Price Prediction

This project predicts housing prices in South Jordan, UT using machine learning models and provides a simple web frontend for predictions.

## Features
- Data preprocessing and feature engineering
- Mainly using RandomForestRegressor and other ML/AI tools
- Hyperparameter tuning
- Error analysis and feature importance
- Flask-based web frontend for predictions

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd houses_model
```

### 2. Create and Activate a Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add your database credentials and other secrets as needed:
```
DB_HOST=your_host
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_db
```

### 5. Run the Model and Frontend
- To run the model and see predictions in the terminal:
  ```
  python3 model.py
  ```
- To start the Flask web frontend:
  ```
  python3 front_end.py
  ```
  Then visit [http://localhost:5000](http://localhost:5000) in your browser.

## File Overview
- `model.py` / `housing_model.py`: Main model logic and prediction methods
- `database_accessor.py`: Database connection and data retrieval
- `front_end.py`: Flask web app for user predictions
- `.env`: Environment variables (not committed)
- `requirements.txt`: Python dependencies
