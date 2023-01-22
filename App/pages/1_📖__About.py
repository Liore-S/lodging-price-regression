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
st.write("This app is created to predict the price of lodging in Bali. The data used in this app is obtained from [**Traveloka.com**](https://www.traveloka.com/). The data is collected in 29 September 2022 with **7.200 data**. Features that are used in this app are: **Star Rating**, **City**, **Type**, **Hotel Facilities**, **Room Facilities**, and **Nearest Point of Interest**.") 

# XGBoost Regressor
st.subheader("XGBoost Regressor")
st.write('Model used are XGBoost Regressor with the following parameters:')
st.write(''' 
    - colsample_bytree = 1
    - learning_rate = 0.1
    - max_depth = 13
    - min_child_weight = 9
    - reg_alpha = 9
    - reg_lambda = 4
    - subsample = 0.5
    ''')

# Model Evaluation
st.subheader("Model Evaluation")
st.write('The model is evaluated using RMSE and R2 Score. The result is as follows:')
st.write('''
    - **RMSE** = 497.645
    - **R2 Score** = 0.842
    ''')
predictionPlotURL = 'https://github.com/Liore-S/lodging-price-regression/blob/main/Picture/grid_prediction.png?raw=true'
predictionPlot = Image.open(requests.get(predictionPlotURL, stream=True).raw)
st.image(predictionPlot, caption='Prediction Plot', width=400)

# Feature Importance
st.subheader("Feature Importance")
st.write('The feature importance is as follows:')
featureImportanceURL = 'https://github.com/Liore-S/lodging-price-regression/blob/main/Picture/feature_importance.png?raw=true'
featureImportance = Image.open(requests.get(featureImportanceURL, stream=True).raw)
st.image(featureImportance, caption='Feature Importance', width=550)