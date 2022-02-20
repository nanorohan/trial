#Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

#Function named dataframe_optimizer is defined. This will reduce space consumption by dataframes.
#Credit - https://www.kaggle.com/rinnqd/reduce-memory-usage and 
#https://www.analyticsvidhya.com/blog/2021/04/how-to-reduce-memory-usage-in-python-pandas/
def dataframe_optimizer(df):
  '''This is a dataframe optimizer'''
  start_mem=np.round(df.memory_usage().sum()/1024**2,2)    
  for col in df.columns:
    col_type=df[col].dtype        
    if col_type!=object:
      c_min=df[col].min()
      c_max=df[col].max()
      if str(col_type)[:3]=='int':
        if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
            df[col]=df[col].astype(np.int8)
        elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
            df[col]=df[col].astype(np.int16)
        elif c_min>np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
            df[col]=df[col].astype(np.int32)
        elif c_min>np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
            df[col]=df[col].astype(np.int64)  
      else:
        if c_min>np.finfo(np.float16).min and c_max<np.finfo(np.float16).max:
            df[col]=df[col].astype(np.float16)
        elif c_min>np.finfo(np.float32).min and c_max<np.finfo(np.float32).max:
            df[col]=df[col].astype(np.float32)
        else:
            df[col]=df[col].astype(np.float64)
  end_mem=np.round(df.memory_usage().sum()/1024**2,2)
  return df

#Import saved data and pickle files
bureau_numerical_merge = dataframe_optimizer(pd.read_csv('bureau_numerical_merge.csv'))
bureau_categorical_merge = dataframe_optimizer(pd.read_csv('bureau_categorical_merge.csv'))
previous_numerical_merge = dataframe_optimizer(pd.read_csv('previous_numerical_merge.csv'))
previous_categorical_merge = dataframe_optimizer(pd.read_csv('previous_categorical_merge.csv'))
filename1 = open('model.pkl', 'rb')
model = pickle.load(filename1)
filename1.close()
filename2 = open('imputer.pkl', 'rb')
imputer = pickle.load(filename2)
filename2.close()
filename3 = open('scaler.pkl', 'rb')
scaler = pickle.load(filename3)
filename3.close()
filename4 = open('imputer_constant.pkl', 'rb')
imputer_constant = pickle.load(filename4)
filename4.close()
filename5 = open('ohe.pkl', 'rb')
ohe = pickle.load(filename5)
filename5.close()

#Define a function to create a pipeline for prediction
def inference(query):
	#Add columns titled DEBT_INCOME_RATIO to application_train
	query_int=query.copy();
	query_int['DEBT_INCOME_RATIO'] = query['AMT_ANNUITY']/query['AMT_INCOME_TOTAL']
	#Add columns titled LOAN_VALUE_RATIO to application_train
	query_int['LOAN_VALUE_RATIO'] = query['AMT_CREDIT']/query['AMT_GOODS_PRICE']
	#Add columns titled LOAN_INCOME_RATIO to application_train
	query_int['LOAN_INCOME_RATIO'] = query['AMT_CREDIT']/query['AMT_INCOME_TOTAL']
	#Merge numerical features from bureau to query
	query_bureau = query_int.merge(bureau_numerical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_BUREAU'))
	#Merge categorical features from bureau to query
	query_bureau = query_bureau.merge(bureau_categorical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_BUREAU'))
	#Drop SK_ID_BUREAU
	query_bureau = query_bureau.drop(columns = ['SK_ID_BUREAU'])
	#Shape of query and bureau data combined
	print('The shape of query and bureau data merged: ', query_bureau.shape)  
	#Merge numerical features from previous_application to query_bureau
	query_bureau_previous = query_bureau.merge(previous_numerical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_PREVIOUS'))
	#Merge categorical features from previous_application to query_bureau
	query_bureau_previous = query_bureau_previous.merge(previous_categorical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_PREVIOUS'))
	#Drop SK_ID_PREV and SK_ID_CURR
	query_bureau_previous = query_bureau_previous.drop(columns = ['SK_ID_PREV'])
	#Shape of query_bureau and previous_application data combined
	print('The shape of query_bureau and previous_application data merged: ', query_bureau_previous.shape)  
	#Drop SK_ID_PREV and SK_ID_CURR
	query_bureau_previous = query_bureau_previous.drop(columns = ['SK_ID_CURR'])
	query_numerical = query_bureau_previous.select_dtypes(exclude=object)
	query_categorical = query_bureau_previous.select_dtypes(include=object)
	query_numerical_imputed = imputer.transform(query_numerical)
	query_numerical_imputed_scaled = scaler.transform(query_numerical_imputed)
	query_categorical_imputed = imputer_constant.transform(query_categorical)
	query_categorical_imputed_ohe = ohe.transform(query_categorical_imputed)
	query_data = np.concatenate((query_numerical_imputed_scaled, query_categorical_imputed_ohe.toarray()), axis = 1)
	predictions = model.predict(query_data)
	return predictions
@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')
   
def main():
	st.title("Client Defaulter Tendency")
    	html_temp = """
    	<div style="background-color:Gainsboro;padding:15px">
    	<h2 style="color:MidnightBlue;text-align:center;">Client Defaulter Tendency</h2>
    	</div>
    	"""
	uploaded_file = st.file_uploader("Choose a file")       
	if uploaded_file is not None:
		query = dataframe_optimizer(pd.read_csv(uploaded_file))
		col_names=query.columns.values.tolist()
		query_prediction = inference(query)
		st.write(query_prediction)
		pred_col=pd.DataFrame(query_prediction, columns = ['Defaulter Tendency'])
		pred_append=pd.concat([query,pred_col], axis=1, ignore_index=True)
		col_names.append('Defaulter Tendency')
		pred_append.columns=col_names
		csv=convert_df(pred_append)
		st.download_button("Press to Download", csv, "Defaulter_predictions.csv", key='download-csv')

if __name__=='__main__':
    main()
