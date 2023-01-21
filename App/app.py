# install dependencies
import pip
import subprocess

subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
pip.main(['install', 'xgboost'])

# import variable from data-processing.py
import streamlit as st
import pandas as pd
import pickle

# Load Variable
hotelFacilities = pickle.load(open('Notebook/Variable/hotelFacilities.pkl', 'rb'))
roomFacilities = pickle.load(open('Notebook/Variable/roomFacilities.pkl', 'rb'))
nearestPoint = pickle.load(open('Notebook/Variable/nearestPointOfInterests.pkl', 'rb'))
colOri = pickle.load(open('Notebook/Variable/col.pkl', 'rb'))

# change hotelFacilities, roomFacilities, nearestPoint to list and delete the first element
hotelFacilities = list(hotelFacilities)[1:]
roomFacilities = list(roomFacilities)[1:]
nearestPoint = list(nearestPoint)[1:]

# Load Model
model = pickle.load(open('./Model/bestModel.pkl', 'rb'))

typeList = ['Type_Hotel', 'Type_Resor', 'Type_Apartemen', 'Type_Vila', 'Type_Guest House', 'Type_Homestay', 'Type_B&B', 'Type_Hostel', 'Type_Camping', 'Type_Lainnya', 'Type_Hotel Kapsul']
cityList = ['City_Badung', 'City_Denpasar', 'City_Gianyar', 'City_Sanur', 'City_Bangli', 'City_Buleleng', 'City_Klungkung', 'City_Tabanan', 'City_Jembrana', 'City_Karangasem']

# Sort the list
typeList.sort()
cityList.sort()

# Collects user input features into dataframe
def user_input_features():
   type = st.sidebar.selectbox('Lodging Type', ('Hotel', 'Resor', 'Apartemen', 'Vila', 'Guest House', 'Homestay',
      'B&B', 'Hostel', 'Camping', 'Lainnya', 'Hotel Kapsul'), 0)
   city = st.sidebar.selectbox('City', ('Badung', 'Denpasar', 'Gianyar', 'Sanur', 'Bangli', 'Buleleng',
      'Klungkung', 'Tabanan', 'Jembrana', 'Karangasem'), 0)
   starRating = st.sidebar.slider('Star Rating', 1, 5, 3)
   size = st.sidebar.slider('Room Size (m2)', 2.0, 90.0, 30.0, 0.1,format='%0.1f')
   occupancy = st.sidebar.slider('Occupancy', 1, 3, 2)
   childOccupancy = st.sidebar.slider('Child Occupancy', 0, 5, 3)
   childAge = st.sidebar.slider('Child Age', 0, 17, 9)
   breakfast = st.sidebar.checkbox('Breakfast Included')
   wifi = st.sidebar.checkbox('Wifi Included')
   refund = st.sidebar.checkbox('Free Cancellation / Refund')
   livingRoom = st.sidebar.checkbox('Living Room')
   hotelFacilitie = st.sidebar.multiselect('Hotel Facilities', (hotelFacilities))
   roomFacilitie = st.sidebar.multiselect('Room Facilities', (roomFacilities))
   pointInterest = st.sidebar.multiselect('Point of Interest', (nearestPoint))
   
   # Handle Checkbox
   breakfast = 1 if breakfast else 0
   wifi = 1 if wifi else 0
   refund = 1 if refund else 0
   livingRoom = 1 if livingRoom else 0

   # Handle Multiselect
   hotelFacilitie = ','.join(hotelFacilitie)
   roomFacilitie = ','.join(roomFacilitie)
   pointInterest = ','.join(pointInterest)

   data = {'size': size,
         'baseOccupancy': occupancy,
         'maxChildOccupancy': childOccupancy,
         'maxChildAge': childAge,
         'isBreakfastIncluded': breakfast,
         'isWifiIncluded': wifi,
         'isRefundable': refund,
         'hasLivingRoom': livingRoom,
         'starRating': starRating,
         'city': city,
         'type': type,
         'hotelFacilities': hotelFacilitie,
         'roomFacilities': roomFacilitie,
         'nearestPoint': pointInterest}
   features = pd.DataFrame(data, index=[0])
   return features

# create function to create dataframe with 0 and 1 value
def create_df(dfOri, df_name, df, prefix):
    value = prefix+dfOri[df_name][0]
    for i in range (0, len(df.columns)):
      column_name = df.columns[i]
      if column_name in value:
         df.loc[0, column_name] = 1
      else:
         df.loc[0, column_name] = 0
    return df

# Sidebar
st.sidebar.header('User Input Features')
df = user_input_features()

# Title
st.title('Bali Lodging Price Estimator')
st.write('This is a web app to estimate the price of lodging in Bali')

# Main Panel
st.write(df)

# create empty dataframe for hotelFacilities, roomFacilities, nearestPoint, with column name from hotelFacilities, roomFacilities, nearestPoint
cityEncode = pd.DataFrame(columns=cityList)
typeEncode = pd.DataFrame(columns=typeList)
roomFacilities_df = pd.DataFrame(columns=roomFacilities)
hotelFacilities_df = pd.DataFrame(columns=hotelFacilities)
nearestPoint_df = pd.DataFrame(columns=nearestPoint)

create_df(df, 'roomFacilities', roomFacilities_df, 'Room_')
create_df(df, 'hotelFacilities', hotelFacilities_df, 'Hotel_')
create_df(df, 'nearestPoint', nearestPoint_df, 'Point_')
create_df(df, 'city', cityEncode, 'City_')
create_df(df, 'type', typeEncode, 'Type_')

df = df.drop(['city', 'type', 'hotelFacilities', 'roomFacilities', 'nearestPoint'], axis=1)
df = pd.concat([df, cityEncode,  typeEncode, roomFacilities_df, hotelFacilities_df, nearestPoint_df], axis=1)
st.write(df)

# change all column data type to unit8 except the first column
# keep the first column as float64
df = df.astype({col: 'float64' for col in df.columns[:1]})
df = df.astype({col: 'uint8' for col in df.columns[1:]})

# check df column order with model column order using colOri, if not the same print False
colOri = colOri[1:]
if (df.columns.tolist() == colOri).all():pass
else:st.warning('The order of the columns is not the same as the model')

# Predict Button
if st.button('Predict'):
   prediction = model.predict(df)
   # Formatting the prediction
   formaString = "Rp{:,.2f}"
   prediction = float(prediction[0])
   formatted_prediction = formaString.format(prediction)
   # print the prediction
   st.metric('Price (IDR)', formatted_prediction)
   # st.write(int(prediction))