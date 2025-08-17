from flask import Flask, request, render_template_string
from housing_model import SouthJordanRandomForestModel

app = Flask(__name__)

# Initialize and train model once at startup
print("Initializing South Jordan Housing Price Model...")
model = SouthJordanRandomForestModel()

# Train the model once at startup (more efficient)
print("Loading data and training model...")
df = model.load_and_preprocess_data()
X, y = model.prepare_features(df)
results = model.train_model(X, y, tune_hyperparameters=True)
error_stats = model.analyze_prediction_errors(results)

print(f"Model ready! Test R² Score: {results['test_r2']:.3f}")

HTML_FORM = """
<!doctype html>
<html>
<head>
    <title>South Jordan, UT Housing Price Prediction</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 2.2em;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        label { 
            font-weight: bold; 
            margin-bottom: 5px; 
            color: #555;
        }
        input, select { 
            padding: 12px; 
            border: 2px solid #ddd; 
            border-radius: 8px; 
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus { 
            outline: none; 
            border-color: #667eea; 
        }
        .predict-btn { 
            grid-column: 1 / -1;
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            font-size: 18px; 
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .predict-btn:hover { 
            transform: translateY(-2px);
        }
        .error { 
            color: #e74c3c; 
            background: #fdf2f2; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 20px 0;
            border-left: 4px solid #e74c3c;
        }
        .result { 
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            color: white; 
            padding: 25px; 
            border-radius: 12px; 
            margin: 20px 0; 
            text-align: center;
        }
        .result h2 {
            margin: 0 0 15px 0;
            font-size: 2.5em;
        }
        .confidence {
            background: #f8f9fa;
            color: #495057;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .model-info {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: #6c757d;
        }
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 South Jordan, UT Housing Price Predictor</h1>
        
        {% if error %}
            <div class="error">
                <strong>❌ Error:</strong> {{ error }}
            </div>
        {% endif %}
        
        <form method="post">
            <div class="form-grid">
                <div class="form-group">
                    <label for="beds">🛏️ Bedrooms</label>
                    <input type="number" name="beds" id="beds" min="1" max="10" value="{{ request.form.beds or '3' }}" required>
                </div>
                
                <div class="form-group">
                    <label for="baths">🛁 Bathrooms</label>
                    <input type="number" name="baths" id="baths" min="1" max="10" step="0.5" value="{{ request.form.baths or '2' }}" required>
                </div>
                
                <div class="form-group">
                    <label for="sqft">📐 Square Feet</label>
                    <input type="number" name="sqft" id="sqft" min="500" max="10000" value="{{ request.form.sqft or '2000' }}" required>
                </div>
                
                <div class="form-group">
                    <label for="days_on_market">📅 Days on Market</label>
                    <input type="number" name="days_on_market" id="days_on_market" min="0" max="365" value="{{ request.form.days_on_market or '0' }}">
                </div>
                
                <button type="submit" class="predict-btn">🔮 Predict Price</button>
            </div>
        </form>
        
        {% if prediction_data %}
            <div class="result">
                <h2>${{ "{:,}".format(prediction_data.predicted_price|int) }}</h2>
                <p>Estimated Market Value</p>
                
                <div class="confidence">
                    <strong>95% Confidence Interval:</strong><br>
                    ${{ "{:,}".format(prediction_data.confidence_interval_95[0]|int) }} - 
                    ${{ "{:,}".format(prediction_data.confidence_interval_95[1]|int) }}
                    <br><br>
                    <small>
                        <strong>Prediction Standard Deviation:</strong> ${{ "{:,}".format(prediction_data.prediction_std|int) }}
                    </small>
                </div>
            </div>
        {% endif %}
        
        <div class="model-info">
            <strong>🤖 Model Information:</strong><br>
            • Random Forest with {{ model_info.n_estimators }} trees<br>
            • Trained on South Jordan, UT housing data<br>
            • Model R² Score: {{ "%.3f"|format(model_info.r2_score) }}<br>
            • Features: beds, baths, sqft, market timing, and engineered variables
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_data = None
    error = None
    
    if request.method == 'POST':
        try:
            # Get form data
            beds = int(request.form['beds'])
            baths = float(request.form['baths'])
            sqft = int(request.form['sqft'])
            days_on_market = int(request.form.get('days_on_market', 0))
            status = request.form.get('status', 'ACTIVE')
            
            print(f"Form data: beds={beds}, baths={baths}, sqft={sqft}, days_on_market={days_on_market}, status={status}")
            
            # Make prediction - this returns a dictionary!
            prediction_result = model.make_prediction(beds, baths, sqft, days_on_market, status)
            print(f"Prediction result: {prediction_result}")
            
            # Extract the prediction data
            prediction_data = {
                'predicted_price': prediction_result['predicted_price'],
                'confidence_interval_95': prediction_result['confidence_interval_95'],
                'prediction_std': prediction_result['prediction_std']
            }
            
            print(f"Final prediction: ${prediction_data['predicted_price']:,.0f}")
            
        except ValueError as e:
            error = f"Invalid input values: {str(e)}"
            print(f"ValueError: {error}")
        except KeyError as e:
            error = f"Missing form field: {str(e)}"
            print(f"KeyError: {error}")
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            print(f"Exception: {error}")
            import traceback
            traceback.print_exc()
    
    # Model info for display
    model_info = {
        'n_estimators': model.model.n_estimators,
        'r2_score': results['test_r2'] if results else 0.0
    }
    
    return render_template_string(HTML_FORM, 
                                prediction_data=prediction_data, 
                                error=error, 
                                model_info=model_info,
                                request=request)

if __name__ == '__main__':
    print("\n🚀 Flask application starting...")
    print("🏠 South Jordan Housing Price Predictor ready!")
    print(f"🎯 Model Performance: R² = {results['test_r2']:.3f}")
    print("📱 Open your browser to http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)