# This is for the data ingestion and here we are using s3 bucket for loading the data.

import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting",True) # it prevents automatic type changes (like float64 to float32) happening in the background without your knowledge.

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from sklearn.preprocessing import LabelEncoder
# from src.connections import s3_connection

def load_params(params_path:str) -> dict:
    "Load the parameter from the Yaml file."

    try:
        # Open the file in read mode
        with open(params_path, 'r') as file:
             # Read and convert YAML file into a Python dictionary
            params = yaml.safe_load(file)
        # Log that parameters were read successfully
        logging.debug("Parameter retrieved from %s",params_path)
        # Return the parameters for further use
        return params
    except FileExistsError:
        logging.error("File %s does not exist",params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("Yaml error: %s",e)
        raise
    except Exception as e:
        logging.error('Unexpected error : %s',e)
        raise

# This function helps to load the data from the csv file
def load_data(data_url:str) -> pd.DataFrame:
    "Load the data from the csv file "
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file: %s",e)
        raise
    except Exception as e:
        logging.error('Unexpected error : %s',e)
        raise

# This function helps to preprocess the data 
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the data "
    try:
        # df.drop(columns=["tweet_id"],inplace=True)
        logging.info("Pre-processing start:----")
        final_df = df[df['Heart Disease'].isin(['Presence','Absence'])] # We are storing the Heart Disease into the final_df
        final_df['Heart Disease'] = final_df['Heart Disease'].replace({'Presence':1,'Absence':0})  # Here we are replacing the Absence review as 0 and Presence review as 1
       
        logging.info("Pre-processing end:----")
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe: %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error during the preprocessing: %s",e)
        raise


# This function is to save the data into the data/raw folder
def save_data(train_data:pd.DataFrame , test_data:pd.DataFrame , data_path:str) ->None:
    "Save the train and test dataset in the raw folder"
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logging.debug("Train and Test data saved to : %s",raw_data_path)
    except Exception as e:
        logging.error("Unexpected error occured while saving the data: %s",e)
        raise

# This is the main function to run the upper code 

def main():
    try:
        params = load_params(params_path='params.yaml')  # this params store the yaml file into the python dictionary format and then we can easily access the variable by using it
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2    #This is the hardcoded method.

        df = load_data(data_url=r'D:\VIKAS\Heart-Disease-Predection\notebooks\Heart_Disease_Prediction.csv')  # This is the alternate method to load the data from the notebook/data.csv
        # s3 = s3_connection.s3_operations("bucket-name","accesskey","secretkey")
        # df = s3.fetch_file_from_s3("data.csv")


        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        logging.info(f"The shape of train_data is {train_data.shape} and the shape of test_data {test_data.shape}")
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s",e)
        print(f"Error: {e}")

if __name__== "__main__":
    main()



