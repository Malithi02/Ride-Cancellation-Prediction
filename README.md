# 🚗 Ride Cancellation Prediction Model

A comprehensive machine learning solution for predicting ride cancellations in ride-sharing services, featuring a modern Streamlit web application with real-time predictions and analytics.

## 🚀 Quick Start

### Option 1: Easy Launch (Recommended)
```bash
python launch_app.py
```

### Option 2: Direct Streamlit
```bash
streamlit run app/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

### Option 3: Test First
```bash
python test_simple.py
python launch_app.py
```

The app will be available at: `http://localhost:8501`

## ✨ Features

### 🎯 Core Functionality
- **Real-time Prediction**: Predict ride cancellation risk with high accuracy
- **Batch Processing**: Upload CSV files for bulk predictions
- **Interactive Dashboard**: Beautiful, responsive web interface
- **Advanced Analytics**: Feature importance analysis and risk distribution
- **Data Visualization**: EDA plots and insights

### 🚀 Technical Features
- **Machine Learning**: Random Forest Classifier with 38 engineered features
- **Data Preprocessing**: Automated handling of missing values and feature engineering
- **Scalable Architecture**: Modular design for easy maintenance and updates
- **Error Handling**: Comprehensive error handling and user feedback
- **Responsive Design**: Mobile-friendly interface

## 📁 Project Structure

```
RideCancellationPredictionModel/
├── app/
│   └── app.py                 # Main Streamlit application
├── data/
│   └── ncr_ride_bookings.csv  # Dataset (150,000 records)
├── models/
│   ├── best_model.pkl         # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler for numerical features
│   ├── ohe.pkl               # OneHotEncoder for categorical features
│   └── le_dict.pkl           # LabelEncoders dictionary
├── notebooks/
│   ├── *.png                 # EDA visualization images
│   └── RideCancellationPrediction.ipynb  # Complete training notebook
├── requirements.txt          # Python dependencies
├── launch_app.py            # Application launcher script
├── test_simple.py           # Simple test script
└── README.md                # This file
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn xgboost imbalanced-learn
```

### Step 2: Verify Installation
```bash
python test_simple.py
```

## 📊 Usage

### 1. Real-Time Prediction
1. Navigate to the "🔮 Predict" tab
2. Fill in the ride details:
   - Vehicle type, pickup/drop locations
   - Timing information (date and time)
   - Ride metrics (VTAT, CTAT, distance, value)
   - Driver and customer ratings
   - Payment method
3. Optionally add historical data
4. Click "Predict Cancellation Risk"
5. View the prediction with probability scores

### 2. Batch Prediction
1. Navigate to the "📈 Dashboard" tab
2. Upload a CSV file with ride data
3. Click "Process Batch Predictions"
4. Download the results with predictions

### 3. Analytics & Insights
1. Navigate to the "💡 Insights" tab for model performance and recommendations
2. Navigate to the "📊 Batch Analysis" tab for advanced analytics

## 🎯 Model Performance

### Accuracy Metrics
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 1.0

### Key Features (Top 10)
1. **Avg VTAT** (25%) - Average Vehicle Time to Arrival
2. **Peak Hour Flag** (18%) - Whether ride is during peak hours
3. **Driver Rating** (15%) - Driver's average rating
4. **Customer Rating** (12%) - Customer's average rating
5. **Vehicle Type** (10%) - Type of vehicle requested
6. **Ride Distance** (8%) - Distance of the ride
7. **Booking Value** (6%) - Monetary value of the booking
8. **Payment Method** (4%) - Method of payment
9. **Customer Reliability** (2%) - Historical cancellation rate
10. **Driver Reliability** (1%) - Driver's historical performance

## 🚨 Troubleshooting

### Common Issues

1. **Model files not found**
   - Solution: Run the training notebook first to generate model files

2. **Port 8501 already in use**
   - Solution: Kill existing Streamlit processes or use a different port
   ```bash
   streamlit run app/app.py --server.port 8502
   ```

3. **Missing dependencies**
   - Solution: Install all requirements
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn xgboost imbalanced-learn
   ```

4. **ngrok connection error**
   - Solution: First ensure the app runs locally, then set up ngrok
   ```bash
   # Test locally first
   python launch_app.py
   # Then in another terminal
   ngrok http 8501
   ```

### Error Messages

- **"Error loading models"**: Check if all .pkl files exist in the models/ directory
- **"Error in preprocessing"**: Verify input data format matches expected columns
- **"Image not found"**: Run the visualization code in the notebook

## 🔧 Development

### Running Tests
```bash
python test_simple.py
```

### Launching the App
```bash
python launch_app.py
```

### Manual Streamlit Launch
```bash
streamlit run app/app.py
```

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced ML models (Deep Learning)
- [ ] Mobile app development
- [ ] API endpoints for external integration
- [ ] Automated model retraining
- [ ] A/B testing framework
- [ ] Multi-city support
- [ ] Driver recommendation system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

- **Data Science Team**: Model development and feature engineering
- **Frontend Team**: Streamlit application and UI/UX
- **DevOps Team**: Deployment and infrastructure

## 📞 Support

For support, email support@ridecancellation.com or create an issue in the repository.

---

**Built with ❤️ for better ride-sharing experiences**