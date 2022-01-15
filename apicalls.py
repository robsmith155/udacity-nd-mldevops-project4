import json
import os

import requests

# Load config
with open('config.json','r') as f:
    config = json.load(f) 

output_path = os.path.join(config['output_model_path'], 'apireturns.txt')

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction?datapath=testdata/testdata.csv').content
response2 = requests.get(f'{URL}/scoring').content
response3 = requests.get(f'{URL}/summarystats?datapath=testdata/testdata.csv').content
response4 = requests.get(f'{URL}/diagnostics?datapath=testdata/testdata.csv').content

#combine all API responses
responses = {
    'Prediction response': json.loads(response1.decode('utf8').replace("'", '"')),
    'Scoring response': json.loads(response2.decode('utf8').replace("'", '"')),
    'Summary stats response': json.loads(response3.decode('utf8').replace("'", '"')),
    'Diagnostics response': json.loads(response4.decode('utf8').replace("'", '"'))
}

#write the responses to your workspace
with open(output_path, 'w') as f:
    f.write(json.dumps(responses))


