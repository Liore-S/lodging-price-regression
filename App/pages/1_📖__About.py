import streamlit as st
import pandas as pd
from PIL import Image
import requests

# Page Config
st.set_page_config(
    page_title="Bali Lodging Prediction", 
    page_icon="https://cdn.icon-icons.com/icons2/1363/PNG/512/travel-holiday-vacation-327_89074.png"
)

# Page Title
st.title("About This App")
st.write("This app is created to predict the price of lodging in Bali. The data used in this app is obtained from [**Traveloka.com**](https://www.traveloka.com/). The data is collected in 29 September 2022 with **7.200 data**. Features that are used in this app are: **Star Rating**, **City**, **Type**, **Hotel Facilities**, **Room Facilities**, and **Nearest Point of Interest**. Target Feature that used in this model is lodging price without **Tax** and **Service Fee**.") 

# XGBoost Regressor
st.subheader("XGBoost Regressor")
st.write('Model used are XGBoost Regressor with the following parameters:')
st.write(''' 
    - colsample_bytree = 0.7
    - learning_rate = 0.1
    - max_depth = 13
    - min_child_weight = 8
    - reg_alpha = 0
    - reg_lambda = 1
    - subsample = 0.5
    ''')

# Model Evaluation
st.subheader("Model Evaluation")
st.write('The model is evaluated using RMSE and R2 Score. The result is as follows:')
st.write('''
    - **RMSE** = 448.558
    - **R2 Score** = 0.871
    ''')
predictionPlotURL = 'Picture/grid_prediction.png'
predictionPlotColorURL = 'Picture/grid_prediction_color.png'
# predictionPlot = Image.open(requests.get(predictionPlotURL, stream=True).raw)
# st.image(predictionPlotURL, caption='Prediction Plot', use_column_width=True)
st.image(predictionPlotColorURL, caption='Prediction Plot Color Coded', use_column_width=True)

# Feature Importance
st.subheader("Feature Importance")
st.write('The feature importance is as follows:')
featureImportanceURL = 'Picture/feature_importance.png'
featureImportanceGainURL = 'Picture/feature_importance_gain.png'
# featureImportance = Image.open(requests.get(featureImportanceURL, stream=True).raw)
# featureImportanceGain = Image.open(requests.get(featureImportanceGainURL, stream=True).raw)
st.write('Feature Importance')
st.image(featureImportanceURL, caption='Feature Importance', use_column_width=True)
st.write('Feature Importance using Gain')
st.image(featureImportanceGainURL, caption='Feature Importance Gain', use_column_width=True)

# Tree Visualization
import pickle
import xgboost as xgb
from xgboost import to_graphviz
# Subheader
st.subheader("Tree Visualization")
st.write('The tree visualization is as follows:')
st.write('Click on the maximize button to see the tree visualization in full screen.')
# Load Model
model = pickle.load(open('./Model/bestModel.pkl', 'rb'))
# Tree Visualization
graphviz = to_graphviz(model, num_trees=model.best_iteration)
st.graphviz_chart(graphviz.source, use_container_width=False)