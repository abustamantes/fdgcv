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
import re

"""
Note: For the anomaly detection algorithms we have to use the following versions of libraries:
sklearn==1.2.2
pandas==1.5.3
category_encoders==2.6.1
"""


app = Flask(__name__)


def input_validation(data):
    # Check if 'eventtype' is either 'status' or 'leave'
    if data.get('eventtype') in ['status', 'leave']:
        print('EventType analysis: LEGIT')
    else:
        print('EventType analysis: MALIGN')

    # check that zone is either bz2452, bz2453, bz2454
    if data.get('zone') in ['bz2452', 'bz2453', 'bz2454','bz2457','bz2458']:
        print('Zone analysis: LEGIT')
    else:
        print('zone analysis: MALIGN')

    # techtype can only be 2 - wifi
    if data.get('techtype') == '2':
        print('TechType Analysis: LEGIT')
    else:
        print('TechType analysis: MALIGN')

    # ensure mac address is in proper hexaddecimal format and length is 56
    mac_address = data.get('mac_address')
    if mac_address and re.match(r'^[A-Fa-f0-9]{56}$', mac_address):
        print('MAC address analysis: LEGIT')
    else:
        print('MAC address analysis: MALIGN')

    # rssi must be int between 0 and 256
    rssi = data.get('rssi')
    if isinstance(rssi, int) and 0 <= rssi <= 256:
        print('RSSI analysis: LEGIT')
    else:
        print('RSSI analysis: MALIGN')

    # epocutc YYY-MM-DD 00:00:00
    epocutc = data.get('epocutc')
    if epocutc and re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', epocutc):
        print('epocutc analysis: LEGIT')
    else:
        print('epocutc analysis: MALIGN')


def data_sanitation(data):
    # For the purpose of this example, data sanitation is just printing the data.
    # In a real application, you would perform cleaning operations here.
    print('Sanitized data:', data)

# Function to create features - Feature engineering part
def create_features(df):
    # Encode 'eventtype' and 'zone' as per the given conditions
    df['eventtype_encoded'] = df['eventtype'].apply(lambda s: 1 if s in ['status', 'leave'] else 0)
    df['epocutc_encoded'] = df['epocutc'].apply(lambda x: 1 if str(x).isdigit() and len(str(x)) == 10 else 0)
    df['zone_encoded'] = df['zone'].apply(lambda s: 1 if s in ['bz2452', 'bz2453', 'bz2454', 'bz2457', 'bz2458'] else 0)

    df['length'] = df['mac_address'].apply(len)
    df['num_special_chars'] = df['mac_address'].apply(lambda s: sum(not c.isalnum() for c in s))
    df['is_valid_hash'] = df['mac_address'].apply(lambda s: 1 if re.match("^[a-f0-9]{56}$", s) else 0)

    df['RSSI_encoded'] = df['RSSI'].apply(lambda x: 1 if (isinstance(x, int) or isinstance(x, float)) and x > -200 else 0)
    df['techtype_encoded'] = df['techtype'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    return df

def anomaly_detection(data):
    # Load the models
    clf = load('final_model_v1.joblib')

    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # Apply the function to your dataframes
    df = create_features(df)

    # Drop the original 'mac_address' column
    df.drop(columns=['mac_address', 'eventtype', 'zone', 'epocutc', 'RSSI', 'techtype'], inplace=True)

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
    print(request.is_json)

    # Parse the incoming JSON request data and return it as a Python dictionary
    content = request.get_json()

    # Print the 'data' dictionary directly
    print("Data received: ", content['data'])

    # Call the input validation function
    #input_validation(content['data'])

    # Call the data sanitation function
    # data_sanitation(content['data'])

    # Call the anomaly detection algorithm
    result = anomaly_detection(content['data'])
    print(result)

    return 'JSON posted'


app.run(host='0.0.0.0', port=80)
