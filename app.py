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
sklearn=1.2.2
pandas=1.5.3
category_encoders=2.6.1
"""


app = Flask(__name__)


def input_validation(data):
    # Check if 'eventtype' is either 'status' or 'leave'
    if data.get('eventtype') in ['status', 'leave']:
        print('EventType analysis: LEGIT')
    else:
        print('EventType analysis: MALIGN')

    # check that zone is either bz2452, bz2453, bz2454
    if data.get('zone') in ['bz2452', 'bz2453', 'bz2454']:
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


def anomaly_detection(data):
    # Load the models
    scaler = load('standard_scaler.joblib')
    he = load('hashing_encoder.joblib')
    clf = load('isolation_forest.joblib')

    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # Define the mapping for eventtype and zone classes
    eventtype_mapping = {'0': '0', 'leave': '1', 'status': '2'}
    zone_mapping = {'bz2454': '0', 'bz2453': '1', 'bz2452': '2', 'bz2457': '3', 'bz2458': '4'}

    df['eventtype'] = df['eventtype'].map(eventtype_mapping)
    df['zone'] = df['zone'].map(zone_mapping)

    # return df

    # Process the DataFrame using the models
    df['RSSI'] = scaler.transform(np.array(df['RSSI']).reshape(-1, 1))

    hashed_features = he.transform(df[['mac_address']])
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
