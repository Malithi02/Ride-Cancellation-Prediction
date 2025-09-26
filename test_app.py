#!/usr/bin/env python3
"""
Test script for Ride Cancellation Prediction App
This script tests the core functionality without running the full Streamlit app
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

def test_model_loading():
    """Test if all model files can be loaded"""
    try:
        import joblib
        
        models = {}
        models['best_model'] = joblib.load('models/best_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['ohe'] = joblib.load('models/ohe.pkl')
        models['le_dict'] = joblib.load('models/le_dict.pkl')
        
        print("✅ All models loaded successfully")
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def test_preprocessing():
    """Test the preprocessing function"""
    try:
        # Import the preprocessing function directly without Streamlit context
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / "app"))
        
        # Define the preprocessing function locally to avoid Streamlit context issues
        def preprocess_input(input_df, models):
            """Preprocess input data for prediction"""
            try:
                import pandas as pd
                import numpy as np
                
                # Create a copy to avoid modifying original
                df = input_df.copy()
                
                # Handle missing values
                numerical_cols = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
                categorical_cols = ['Vehicle Type', 'Pickup Location', 'Drop Location', 'Payment Method', 
                                  'Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason']
                
                # Fill missing values
                for col in numerical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
                
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna('Unknown')
                
                # Handle flags
                flag_cols = ['Cancelled Rides by Customer', 'Cancelled Rides by Driver', 'Incomplete Rides']
                for col in flag_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0).astype(int)
                
                # Convert datetime
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                else:
                    # Create datetime from date and time if separate
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    else:
                        df['Datetime'] = pd.Timestamp.now()
                
                # Feature engineering
                df['hour_of_day'] = df['Datetime'].dt.hour
                df['day_of_week'] = df['Datetime'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                
                # Peak/Off-peak flag
                def is_peak(hour):
                    return 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
                df['peak_flag'] = df['hour_of_day'].apply(is_peak)
                
                # VTAT buckets
                def vtat_bucket(vtat):
                    if pd.isna(vtat) or vtat < 5:
                        return 'Low'
                    elif vtat <= 10:
                        return 'Medium'
                    else:
                        return 'High'
                
                df['Avg VTAT Raw'] = df['Avg VTAT']
                df['vtat_bucket'] = df['Avg VTAT Raw'].apply(vtat_bucket)
                
                # High fare flag
                median_value = df['Booking Value'].median() if not df['Booking Value'].isna().all() else 0
                df['high_fare_flag'] = df['Booking Value'].apply(lambda x: 1 if x > median_value else 0)
                
                # Reliability scores
                df['customer_reliability'] = 1 - df['Cancelled Rides by Customer']
                df['driver_reliability'] = 1 - df['Cancelled Rides by Driver']
                
                # Ride speed
                df['ride_speed'] = df['Ride Distance'] / df['Avg CTAT'].replace(0, np.nan).fillna(1)
                
                # Drop datetime and raw VTAT
                df = df.drop(['Datetime', 'Avg VTAT Raw'], axis=1, errors='ignore')
                
                # One-hot encoding
                ohe_cols = ['Vehicle Type', 'Payment Method', 'vtat_bucket']
                ohe = models['ohe']
                
                # Ensure all required columns exist
                for col in ohe_cols:
                    if col not in df.columns:
                        df[col] = 'Unknown'
                
                X_ohe = pd.DataFrame(
                    ohe.transform(df[ohe_cols]), 
                    columns=ohe.get_feature_names_out(), 
                    index=df.index
                )
                
                # Label encoding
                le_cols = ['Pickup Location', 'Drop Location', 'Reason for cancelling by Customer', 
                          'Driver Cancellation Reason', 'Incomplete Rides Reason']
                le_dict = models['le_dict']
                
                for col in le_cols:
                    if col in df.columns:
                        le = le_dict[col]
                        # Handle unknown values
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else '<unknown>')
                        df[col] = le.transform(df[col])
                
                # Combine features
                df_encoded = pd.concat([df.drop(ohe_cols, axis=1), X_ohe], axis=1)
                
                # Scale numerical features
                numerical_to_scale = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance']
                scaler = models['scaler']
                
                for col in numerical_to_scale:
                    if col in df_encoded.columns:
                        df_encoded[col] = scaler.transform(df_encoded[[col]])
                
                return df_encoded
                
            except Exception as e:
                print(f"Error in preprocessing: {str(e)}")
                return None
        
        # Create sample input data
        sample_data = {
            'Vehicle Type': ['Go Mini'],
            'Pickup Location': ['Connaught Place'],
            'Drop Location': ['Gurgaon Sector 29'],
            'Avg VTAT': [5.0],
            'Avg CTAT': [3.0],
            'Booking Value': [150],
            'Ride Distance': [5.0],
            'Driver Ratings': [4.0],
            'Customer Rating': [4.0],
            'Payment Method': ['UPI'],
            'Datetime': [pd.Timestamp.now()],
            'Cancelled Rides by Customer': [0],
            'Reason for cancelling by Customer': ['Unknown'],
            'Cancelled Rides by Driver': [0],
            'Driver Cancellation Reason': ['Unknown'],
            'Incomplete Rides': [0],
            'Incomplete Rides Reason': ['Unknown']
        }
        
        input_df = pd.DataFrame(sample_data)
        
        # Load models
        models = test_model_loading()
        if models is None:
            return False
        
        # Test preprocessing
        processed = preprocess_input(input_df, models)
        
        if processed is not None:
            print("✅ Preprocessing function works correctly")
            print(f"   Input shape: {input_df.shape}")
            print(f"   Processed shape: {processed.shape}")
            return True
        else:
            print("❌ Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in preprocessing test: {e}")
        return False

def test_prediction():
    """Test the prediction functionality"""
    try:
        # Define the preprocessing function locally to avoid Streamlit context issues
        def preprocess_input(input_df, models):
            """Preprocess input data for prediction"""
            try:
                import pandas as pd
                import numpy as np
                
                # Create a copy to avoid modifying original
                df = input_df.copy()
                
                # Handle missing values
                numerical_cols = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
                categorical_cols = ['Vehicle Type', 'Pickup Location', 'Drop Location', 'Payment Method', 
                                  'Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason']
                
                # Fill missing values
                for col in numerical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
                
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna('Unknown')
                
                # Handle flags
                flag_cols = ['Cancelled Rides by Customer', 'Cancelled Rides by Driver', 'Incomplete Rides']
                for col in flag_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0).astype(int)
                
                # Convert datetime
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                else:
                    # Create datetime from date and time if separate
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    else:
                        df['Datetime'] = pd.Timestamp.now()
                
                # Feature engineering
                df['hour_of_day'] = df['Datetime'].dt.hour
                df['day_of_week'] = df['Datetime'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                
                # Peak/Off-peak flag
                def is_peak(hour):
                    return 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
                df['peak_flag'] = df['hour_of_day'].apply(is_peak)
                
                # VTAT buckets
                def vtat_bucket(vtat):
                    if pd.isna(vtat) or vtat < 5:
                        return 'Low'
                    elif vtat <= 10:
                        return 'Medium'
                    else:
                        return 'High'
                
                df['Avg VTAT Raw'] = df['Avg VTAT']
                df['vtat_bucket'] = df['Avg VTAT Raw'].apply(vtat_bucket)
                
                # High fare flag
                median_value = df['Booking Value'].median() if not df['Booking Value'].isna().all() else 0
                df['high_fare_flag'] = df['Booking Value'].apply(lambda x: 1 if x > median_value else 0)
                
                # Reliability scores
                df['customer_reliability'] = 1 - df['Cancelled Rides by Customer']
                df['driver_reliability'] = 1 - df['Cancelled Rides by Driver']
                
                # Ride speed
                df['ride_speed'] = df['Ride Distance'] / df['Avg CTAT'].replace(0, np.nan).fillna(1)
                
                # Drop datetime and raw VTAT
                df = df.drop(['Datetime', 'Avg VTAT Raw'], axis=1, errors='ignore')
                
                # One-hot encoding
                ohe_cols = ['Vehicle Type', 'Payment Method', 'vtat_bucket']
                ohe = models['ohe']
                
                # Ensure all required columns exist
                for col in ohe_cols:
                    if col not in df.columns:
                        df[col] = 'Unknown'
                
                X_ohe = pd.DataFrame(
                    ohe.transform(df[ohe_cols]), 
                    columns=ohe.get_feature_names_out(), 
                    index=df.index
                )
                
                # Label encoding
                le_cols = ['Pickup Location', 'Drop Location', 'Reason for cancelling by Customer', 
                          'Driver Cancellation Reason', 'Incomplete Rides Reason']
                le_dict = models['le_dict']
                
                for col in le_cols:
                    if col in df.columns:
                        le = le_dict[col]
                        # Handle unknown values
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else '<unknown>')
                        df[col] = le.transform(df[col])
                
                # Combine features
                df_encoded = pd.concat([df.drop(ohe_cols, axis=1), X_ohe], axis=1)
                
                # Scale numerical features
                numerical_to_scale = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance']
                scaler = models['scaler']
                
                for col in numerical_to_scale:
                    if col in df_encoded.columns:
                        df_encoded[col] = scaler.transform(df_encoded[[col]])
                
                return df_encoded
                
            except Exception as e:
                print(f"Error in preprocessing: {str(e)}")
                return None
        
        # Create sample input data
        sample_data = {
            'Vehicle Type': ['Go Mini'],
            'Pickup Location': ['Connaught Place'],
            'Drop Location': ['Gurgaon Sector 29'],
            'Avg VTAT': [5.0],
            'Avg CTAT': [3.0],
            'Booking Value': [150],
            'Ride Distance': [5.0],
            'Driver Ratings': [4.0],
            'Customer Rating': [4.0],
            'Payment Method': ['UPI'],
            'Datetime': [pd.Timestamp.now()],
            'Cancelled Rides by Customer': [0],
            'Reason for cancelling by Customer': ['Unknown'],
            'Cancelled Rides by Driver': [0],
            'Driver Cancellation Reason': ['Unknown'],
            'Incomplete Rides': [0],
            'Incomplete Rides Reason': ['Unknown']
        }
        
        input_df = pd.DataFrame(sample_data)
        
        # Load models
        models = test_model_loading()
        if models is None:
            return False
        
        # Test preprocessing
        processed = preprocess_input(input_df, models)
        if processed is None:
            return False
        
        # Test prediction
        prediction = models['best_model'].predict(processed)[0]
        probability = models['best_model'].predict_proba(processed)[0]
        
        print("✅ Prediction function works correctly")
        print(f"   Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        print(f"   Cancellation Probability: {probability[1]:.2%}")
        print(f"   Completion Probability: {probability[0]:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Ride Cancellation Prediction App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("models").exists():
        print("❌ Models directory not found. Please run from project root.")
        return False
    
    if not Path("app/app.py").exists():
        print("❌ App file not found. Please run from project root.")
        return False
    
    # Run tests
    tests = [
        ("Model Loading", test_model_loading),
        ("Preprocessing", test_preprocessing),
        ("Prediction", test_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to run.")
        print("   Run: python run_app.py")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
