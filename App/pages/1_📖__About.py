from xgboost import to_graphviz
import pickle
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Bali Lodging Prediction",
    page_icon=":hotel:"
)

# Page Title
st.title("About This App")
st.write(
    "This app is created to predict the price of lodging in Bali. The data used in this app is obtained from [**Traveloka.com**](https://www.traveloka.com/). The data is collected in 29 September 2022 with **7.200 data**. Features that are used in this app are: **Star Rating**, **City**, **Type**, **Hotel Facilities**, **Room Facilities**, and **Nearest Point of Interest**. Target Feature that used in this model is lodging price without **Tax** and **Service Fee**.")

# XGBoost Regressor
params = pickle.load(open('./Notebook/Variable/bestParams.pkl', 'rb'))
st.subheader("XGBoost Regressor")
st.write('Model used are XGBoost Regressor with the following parameters:')
st.write(f''' 
    - colsample_bytree = {params['colsample_bytree']}
    - learning_rate = {params['learning_rate']}
    - max_depth = {params['max_depth']}
    - min_child_weight = {params['min_child_weight']}
    - reg_alpha = {params['reg_alpha']}
    - reg_lambda = {params['reg_lambda']}
    - subsample = {params['subsample']}
    ''')

# Model Evaluation
score = pickle.load(open('./Notebook/Variable/score.pkl', 'rb'))
st.subheader("Model Evaluation")
st.write('The model is evaluated using RMSE and R2 Score. The result is as follows:')
st.write(f'''
    - **RMSE** = {score['RMSE']}
    - **R2 Score** = {score['R2']}
    ''')
predictionPlotURL = 'Picture/grid_prediction.png'
predictionPlotColorURL = 'Picture/grid_prediction_color.png'
# predictionPlot = Image.open(requests.get(predictionPlotURL, stream=True).raw)
# st.image(predictionPlotURL, caption='Prediction Plot', use_column_width=True)
st.image(predictionPlotColorURL,
         caption='Prediction Plot Color Coded', use_column_width=True)

# Feature Importance
st.subheader("Feature Importance")
st.write('The feature importance is as follows:')
featureImportanceURL = 'Picture/feature_importance.png'
featureImportanceGainURL = 'Picture/feature_importance_gain.png'
# featureImportance = Image.open(requests.get(featureImportanceURL, stream=True).raw)
# featureImportanceGain = Image.open(requests.get(featureImportanceGainURL, stream=True).raw)
st.write('Feature Importance')
st.image(featureImportanceURL, caption='Feature Importance',
         use_column_width=True)
st.write('Feature Importance using Gain')
st.image(featureImportanceGainURL,
         caption='Feature Importance Gain', use_column_width=True)

# Tree Visualization
# Subheader
st.subheader("Tree Visualization")
st.write('The tree visualization is as follows:')
st.write('Click on the maximize button to see the tree visualization in full screen.')
# Load Model
model = pickle.load(open('./Model/bestModel.pkl', 'rb'))
# Tree Visualization
graphviz = to_graphviz(model, num_trees=model.best_iteration)
st.graphviz_chart(graphviz.source, use_container_width=False)
