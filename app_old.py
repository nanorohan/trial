#Import libraries
import pandas as pd
import csv
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import base64
from PIL import Image

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
	pred_cat=[]
	for i in range(len(predictions)):
		if predictions[i]==0:
			pred_cat.append("Low")
		else:
			pred_cat.append("High")
	applicant_no=query['SK_ID_CURR'].copy()
	#applicant_no['Defaulting Tendency']=pred_cat
	pred_df=pd.DataFrame(pred_cat, columns = ['Defaulting Tendency'])
	pred_out=pd.concat([applicant_no,pred_df], axis=1, ignore_index=False)
	return pred_out
@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')
   
def main():
	st.set_page_config(
	     page_title="Loan Defaulter Predictor",
	     page_icon="????",
	     layout="wide",
	     initial_sidebar_state="collapsed",
	     menu_items={
		 'Get Help': 'https://www.extremelycoolapp.com/help',
		 'Report a bug': "https://github.com/nanorohan",
		 'About': "This"
	     }
	 )	
	header_pic = Image.open('loan_alt.jpg')
	st.image(header_pic)
	home_page = f"""
        <div style="display:flex;justify-content:space-between;background:#3E4248;padding:10px;border-radius:5px;margin:5px;">
            <div style="float:justify; text-align: justify; width:100%; background:#3E4248; padding:10px; border-radius:5px; margin:5px;">
                <p style="text-align: justify; color:#FFF8DC; line-height: 1.35;font-size: 23px; font-family:Playfair;">
		    <b>Loan Defaulter Predictor helps you identify potential defaulters.</b>
		    <br>
		    Given a loan application of a potential or existing client at Home Credit, this app "predicts" whether 
		    the client will be able to repay the loan or not.
		    <br>
		    Applicants deemed capable of repaying the loan shall be labelled <b>Low</b> under defaulting tendency and those deemed as incapable 
		    shall be labelled as <b>High</b>.    
                </p>    
            </div>
        </div>
	"""
	st.write(home_page, unsafe_allow_html=True)

	# Side bar portion of code
	overview_desc = f"""
        <div style="display:flex;justify-content:space-between;background:#FFEBCD;padding:10px;border-radius:5px;margin:5px;">
            <div style="float:justify; text-align: justify; width:100%; background:#FFEBCD; padding:10px; border-radius:5px; margin:5px;">
                <p style="text-align: justify; color:#8B0000; line-height: 1.35;font-size: 23px; font-family:Playfair;">
		<h2 style="color:#000059; font-family:Playfair;">Why this app?</h2>
		<p style="color:#8B0000; text-align: justify; line-height: 1.35;font-size: 23px; font-family:Playfair;">
		Post-pandemic world has disrupted many aspects of life including financial requirements. 
		Many are resorting to loans to ensure basic subsistence. Some struggle to get loans due to insufficient or 
		non-existent credit histories. And, unfortunately, such a population is often taken advantage of by unscrupulous lenders. 
		Conversely, lending institutions need to ensure very low credit delinquency rates to stay profitable and provide loans to 
		worthy applicants.
		<br>
 		An objective model helps the lending agencies disburse prudent loans. This system helps process and disburse loans faster, 
		increase profit and client base  as well as possibly protect genuine debtors from predatory lenders.
		<br>
		Thus, this app.
		<br>
		<br>
		</p>
                <p style="text-align: justify; color:#8B0000; line-height: 1.35;font-size: 23px; font-family:Playfair;">
		<h2 style="color:#000059; font-family:Playfair;">Behind the Secene</h2>
		<p style="color:#8B0000; text-align: justify; line-height: 1.35;font-size: 23px; font-family:Playfair;">
		This app is powered by Machine Learning! It uses a 
		<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">Gradient Boosting Classifier model from SKlearn library</a>
		() trained on Kaggle's <a href="https://www.kaggle.com/c/home-credit-default-risk/overview">Home Credit Default Risk dataset</a>. 
		I have used Streamlit & deployed this app on Heroku.
		A complete repository of this app right from data preprocessing upto the deployment can be viewed on my <a href="https://github.com/nanorohan">GitHub page</a>.
		Feedback and ideas may be directed <a href="mailto:nanorohan@gmail.com">here</a>.
		</p>		
	    </div>
	</div>
    	"""
	st.sidebar.write(overview_desc, unsafe_allow_html=True)

	html_template = """
	<html>
	<body>
	<style> 
	#rcorners2 {border-radius: 20px; border: 3px solid #c83349; background: #1E323B; padding: 10px; color:tomato; 
	font-size:23px; font-family:Playfair; text-align:center;
	}	
	</style>
	<p id="rcorners2" >Please adhere to the template below to fill in applicant details for predicting defaulting tendency</p>
	<br>
     	</body>
     	</html>	
	"""
	st.markdown(html_template,unsafe_allow_html=True)
	
	with open("applicants_details_template.csv") as template_file:
		st.download_button("Download Applicant details template", template_file, "applicants_details_template.csv", key='download-csv')	
	print("\n\n\n\n")
	html_uploader = """
	<html>
	<body>
	<meta name="viewport" content="width=device-width, initial-scale=1">
  	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css">
	<style> 
	p {
	  font-size: 23px;
	  color: #151B54;
	  font-family: Playfair, bold;
	  text-align: center;
	}	
	#rcorners1 {border-radius: 20px; border: 3px solid #c83349; background: #ffcc5c; padding: 10px;
	}	
	</style>
	<br>
	<p id="rcorners1" >Upload applicant details in required format</p>
	</body>
     	</html>
	"""	
	st.markdown(html_uploader,unsafe_allow_html=True)
	uploaded_file = st.file_uploader(" ")       
	if uploaded_file is not None:
		query = dataframe_optimizer(pd.read_csv(uploaded_file))
		col_names=query.columns.values.tolist()
		query_prediction = inference(query)
		query_pred=pd.DataFrame(query_prediction) 
		query_pred.columns = ['Applicant ID', 'Defaulting Tendency']
		st.dataframe(query_pred)
		#pred_col=pd.DataFrame(query_prediction, columns = ['Defaulter Tendency'])
		pred_append=pd.concat([query,query_pred['Defaulting Tendency']], axis=1, ignore_index=False)
		#col_names.append('Defaulting Tendency')
		#pred_append.columns=col_names
		csv=convert_df(pred_append)

		html_file_dl = """
		<html>
		<body>
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css">
		<style> 
		p {
		  font-size: 23px;
		  color: #151B54;
		  font-family: Playfair, bold;
		  text-align: center;
		}	
		#rcorners1 {border-radius: 20px; border: 3px solid #c83349; background: #ffcc5c; padding: 10px;
		}	
		</style>
		<br>
		<p id="rcorners1" >For downloading the predictions appended to the applicant details, click below link</p>
		</body>
		</html>
		"""
		st.markdown(html_file_dl,unsafe_allow_html=True)
		st.download_button("Press to Download", csv, "Defaulter_predictions.csv", key='download-csv')

if __name__=='__main__':
    main()
