import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Fetal Health Detection",
    page_icon="👶",
    layout="wide"
)

# Define absolute model paths dictionary
MODEL_PATHS = {
    "AdaBoost": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\adaboost_model.pkl",
    "Decision Tree": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\decision_tree_model.pkl",
    "Gaussian NB": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\gaussian_nb_model.pkl",
    "K-Nearest Neighbors": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\kneighbors_model.pkl",
    "Logistic Regression": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\logistic_regression_model.pkl",
    "PSO Weighted Voting": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\pso_weighted_voting_model.pkl",
    "Soft Voting": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\soft_voting_model.pkl",
    "Stacking": r"C:\Users\shweta\Desktop\Shweta\Major project\fetal_health_app\stacking_model.pkl"
}

# Path to dataset
DATA_PATH = r"C:\Users\shweta\Desktop\Shweta\Major project\Dataset\balanced_ga_dataset_normalized.csv"

# Sidebar model selector
st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a model", list(MODEL_PATHS.keys()))

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATHS[selected_model_name])

# Load dataset
@st.cache_data
def load_dataset(data_path):
    return pd.read_csv(data_path)

data = load_dataset(DATA_PATH)

# Title
st.title('Fetal Health Classification')
st.markdown("""
    This app predicts fetal health using Cardiotocogram (CTG) features. 
    Adjust the sliders below to simulate input values.
""")

# Get the correct features for the model
def get_model_features(model):
    # Try different attribute names that scikit-learn models might use
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    elif hasattr(model, 'feature_names'):
        return model.feature_names
    # If model is voting classifier or other ensemble
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Try to get features from the first base estimator
        base_model = model.estimators_[0]
        if hasattr(base_model, 'feature_names_in_'):
            return base_model.feature_names_in_
    
    # Fallback: load a sample from the dataset and see which features are needed
    try:
        X_sample = data.drop('fetal_health', axis=1) if 'fetal_health' in data.columns else data
        model.predict(X_sample.iloc[[0]])
        return X_sample.columns.tolist()
    except:
        # If all else fails, print a warning and return all features
        st.warning("Could not determine model features. Using all input features.")
        return data.drop('fetal_health', axis=1).columns.tolist() if 'fetal_health' in data.columns else data.columns.tolist()

# Try to get model features
if model is not None:
    try:
        model_features = get_model_features(model)
        st.info(f"Model expects {len(model_features)} features")
    except Exception as e:
        st.error(f"Error determining model features: {e}")
        model_features = data.drop('fetal_health', axis=1).columns.tolist() if 'fetal_health' in data.columns else data.columns.tolist()
else:
    model_features = data.drop('fetal_health', axis=1).columns.tolist() if 'fetal_health' in data.columns else data.columns.tolist()

# Layout input columns
col1, col2 = st.columns(2)
inputs = {}

# Only display the features that the model actually uses
feature_names = [feature for feature in model_features if feature in data.columns]

for i, feature in enumerate(feature_names):
    min_val = float(data[feature].min())
    max_val = float(data[feature].max())
    mean_val = float(data[feature].mean())

    with (col1 if i % 2 == 0 else col2):
        if np.issubdtype(data[feature].dtype, np.integer):
            inputs[feature] = st.slider(f"{feature}", int(min_val), int(max_val), int(mean_val))
        else:
            inputs[feature] = st.slider(f"{feature}", float(min_val), float(max_val), float(mean_val))

# Prediction
st.subheader('Prediction')
if st.button("Predict Fetal Health"):
    if model is not None:
        try:
            # Create input dataframe with only the required features
            input_df = pd.DataFrame([inputs])
            
            # Debug information (can be removed in production)
            with st.expander("Debug Information"):
                st.write(f"Input features: {input_df.columns.tolist()}")
                st.write(f"Model features: {model_features}")
                st.write(f"Features in input but not in model: {set(input_df.columns) - set(model_features)}")
                st.write(f"Features in model but not in input: {set(model_features) - set(input_df.columns)}")
            
            # Ensure only model features are used and in correct order
            for feature in model_features:
                if feature not in input_df.columns:
                    # If a required feature is missing, use mean value from dataset
                    if feature in data.columns:
                        input_df[feature] = data[feature].mean()
                    else:
                        st.error(f"Missing feature '{feature}' required by model but not in dataset.")
                        raise ValueError(f"Missing feature: {feature}")
            
            # Reorder columns to match model's expected feature order
            input_df = input_df[model_features]
            
            # Predict
            prediction = model.predict(input_df)[0]

            health_status = {
                1: 'Normal',
                2: 'Suspect',
                3: 'Pathological'
            }

            st.success(f"Predicted Fetal Health: **{health_status[prediction]}** using **{selected_model_name}**")

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_df)[0]
                st.write("Probability Scores:")
                prob_df = pd.DataFrame({
                    "Health Status": list(health_status.values()),
                    "Probability": probs
                })
                for i, prob in enumerate(probs):
                    st.write(f"{health_status[i+1]}: {prob:.2%}")
                
                st.bar_chart(prob_df.set_index("Health Status"))
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Detailed traceback:", exception=e)
    else:
        st.error("Model could not be loaded. Please select a different model.")

# Sidebar info
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Select a model from the dropdown
2. Adjust input values using sliders
3. Click 'Predict Fetal Health' to get results
""")

st.sidebar.title("About")
st.sidebar.info("""
This app uses ML models to predict fetal health based on CTG data from the normalized dataset.
""")