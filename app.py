from flask import Flask
from flask import request
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from category_encoders import HashingEncoder
import numpy as np
from joblib import dump


app = Flask(__name__)


def input_validation(data):
    # Check if 'eventtype' is either 'status' or 'leave'
    if data.get('eventtype') in ['status', 'leave']:
        print('Result of the analysis: Legit data')
    else:
        print('Result of the analysis: Malign data')


def data_sanitation(data):
    # For the purpose of this example, data sanitation is just printing the data.
    # In a real application, you would perform cleaning operations here.
    print('Sanitized data:', data)

def anomaly_detection(data):
    # Load the models
    scaler = load('D:/UBamberg/UBamberg-Topic-IV-and-V/standard_scaler.joblib')
    he = load('D:/UBamberg/UBamberg-Topic-IV-and-V/hashing_encoder.joblib')
    clf = load('D:/UBamberg/UBamberg-Topic-IV-and-V/isolation_forest.joblib')

    # Create a DataFrame from the input data
    df = pd.DataFrame([data])
    #print(df['eventtype'])
    #print(df['epocutc'])
    #print(df['zone'])
    #print(df['mac_address'])
    #print(df['RSSI'])
    #print(df['techtype'])

    # Define the mapping for eventtype and zone classes
    eventtype_mapping = {'0': '0', 'leave': '1', 'status': '2'}
    zone_mapping = {'bz2454': '0', 'bz2453': '1', 'bz2452': '2', 'bz2457': '3', 'bz2458': '4'}

    df['eventtype'] = df['eventtype'].map(eventtype_mapping)
    df['zone'] = df['zone'].map(zone_mapping)

    #return df

    # Process the DataFrame using the models
    df['RSSI'] = scaler.transform(np.array(df['RSSI']).reshape(-1, 1))

    print(df['eventtype'])
    print(df['epocutc'])
    print(df['zone'])
    print(df['mac_address'])
    print(type(df['mac_address']))

    print(df['RSSI'])
    print(df['techtype'])

    print(he)  # Add this to check the status of HashingEncoder
    print(df['mac_address'].dtypes)  # Add this to check the datatype of mac_address
    print(df['mac_address'].isnull().any())  # Add this to check if there's any null value

    hashed_features = he.transform(df[['mac_address']])

    print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

    hashed_features.columns = [f'mac_address_{i}' for i in hashed_features.columns]
    df = pd.concat([df, hashed_features], axis=1)
    df = df.drop(columns=['mac_address'])
  
    pred = clf.predict(df)

    # The model will output -1 for anomalies/outliers and 1 for inliers.
    df['anomaly'] = pred

    # Check for anomaly
    if df['anomaly'].values[0] == -1:
        return "The request is malign"
    else:
        return "The request is benign"

@app.route('/postjson', methods=['POST'])
def postJsonHandler():
    print("\nNew request: ")
    print("*************")
    # Confirm that content is of JSON type
    #print(request.is_json)

    # Parse the incoming JSON request data and return it as a Python dictionary
    content = request.get_json()

    # Print the 'data' dictionary directly
    print("Data received: ",content['data'])

    # Call the input validation function
    #input_validation(content['data'])

    # Call the data sanitation function
    #data_sanitation(content['data'])

    # Call the anomaly detection algorithm
    result = anomaly_detection(content['data'])
    print(result)

    return 'JSON posted'


app.run(host='0.0.0.0', port=5100)
