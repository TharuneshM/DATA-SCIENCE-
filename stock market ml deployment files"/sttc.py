from datetime import date
from nsepy import get_history
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yahooFinance

import streamlit as st


#TITLE 
st.title('STOCK MARKET DATA ANALSYIS USING LSTM')


#EXTRACTING DATA
symbol="MRF.NS"
start=date(2012,1,1)
end=date(2023,3,1)
user_ip=st.text_input('enter stock ticker','MRF')
GetMRFInformation = yahooFinance.Ticker("MRF.NS")
df=GetMRFInformation.history(start=start,end=end)


#DISPAYING THE DESCRIPTIVE STATISTICS OF DATA
st.subheader('Data from 2013 - 2023')
st.write(df.describe())


#VISUALATION OF CLOSE PRICE
st.subheader('CLOSING PRICE')
fig=plt.figure(figsize=(12,8))
plt.plot(df.Close)
st.pyplot(fig)


#SCALING THE DATA
df1=df.reset_index()['Close']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


#SPLITING THE DATA
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


#LOADING THE MODEL
from keras.models import load_model
model=load_model('C:\\Users\\Tharun\\Desktop\\PROJECT ML\\stockmarket forecasting\\model_h5.h5')

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


#TESTING THE DATA
Y=ytest.reshape(-1, 1)
y_test=scaler.inverse_transform(Y)


#VISULAZING THE ACTUAL VALUES VS PREDICTED VALUES
st.subheader('ACTUAL VALUES vs PREDICTED VALUES')
fig=plt.figure(figsize=(12,8))
plt.plot(y_test)
plt.plot(test_predict,color="red")
st.pyplot(fig)


#PREDICTING THE NEXT 30 DAYS CLOSING PRICE
x_input=test_data[681:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=146
i=0
while(i<30):
    
    if(len(temp_input)>146):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)
day_new=np.arange(1,147)
day_pred=np.arange(147,177)


# #VISUALIZING THE FORECASTED VALUES
st.subheader('FORECASTING CLOSE PRICE FOR NEXT 30 DAYS')
fig=plt.figure(figsize=(12,8))
plt.plot(day_new,scaler.inverse_transform(df1[2608:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output),color="red")
st.pyplot(fig)


#FORECASTED VALUE FOR 30 DAYS
#st.subheader('FORECASTING THE CLOSING PRICE FOR THE NEXT 30 DAYS')
forecasted_vals=scaler.inverse_transform(lst_output)
st.subheader('FORECASTED VALUES')
#st.write(forecasted_vals)




from datetime import date, timedelta
import numpy as np

# Set default start and end dates
start_date = date(2023,3,2)
end_date = start_date + timedelta(days=30)

# Create date input range widget
date_range = st.date_input("DATE RANGE",value=(start_date, end_date),min_value=None,max_value=None,key=None)

# Get start and end dates from the widget
start_date = date_range[0]
end_date = date_range[1]

# Calculate the number of days between the start and end dates
delta = end_date - start_date
num_days = delta.days + 1


# Display the forecast values
#for i, val in enumerate(day_pred):
    #date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    #st.write(date, ":", val)


# Display the forecast values and the forecast dates

arr=np.empty([30],dtype='object')
index=0
for i, val in enumerate(day_pred):
    date=(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    arr[index]=date
    index+=1
st.write(np.append(forecasted_vals,arr.reshape(30,1),axis=1))






