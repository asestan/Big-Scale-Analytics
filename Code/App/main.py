# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]
# [START gae_python3_app]
import datetime, sys, re, pickle

from flask import Flask, render_template, request
from google.cloud import automl_v1
from google.api_core.client_options import ClientOptions

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

#Calling Google AutoML API and returning the prediction
def get_automl_prediction(input, model_name):
  options = ClientOptions(api_endpoint='automl.googleapis.com')
  prediction_client = automl_v1.PredictionServiceClient(client_options=options)

  payload = {'text_snippet': {'content': input, 'mime_type': 'text/plain'} }
  params = {}
  automl_request = automl_v1.PredictRequest(name=model_name, payload=payload, params=params)
  
  automl_response = prediction_client.predict(automl_request)
  return automl_response  # waits until request is returned

#Cleaning the sentence (removal of punctuations)
def sentence_cleaning(input):
    clean_input = re.sub(r'[^\w\s]', '', input)
    return clean_input

@app.route('/', methods=['GET','POST'])
def root():
    form_data = ""
    difficulty = ""
    result_list = [0,0,0,0,0,0]
    given_sentence = ""
    if request.method == 'POST':
        #Getting and cleaning data
        form_data = request.form.get('textinput')
        clean_form_data = sentence_cleaning(form_data)

        #Predictions
        prediction = get_automl_prediction(clean_form_data, "projects/79067930854/locations/us-central1/models/TCN6320175355386658816")
        difficulty = prediction.payload[0].display_name
        given_sentence = "The sentence \"" + form_data +"\" is labelled as "
        for label in prediction.payload:
            if label.display_name =="A1":
                result_list[0] = round(label.classification.score*100, 1)
            elif label.display_name =="A2":
                result_list[1] = round(label.classification.score*100, 1)
            elif label.display_name =="B1":
                result_list[2] = round(label.classification.score*100, 1)
            elif label.display_name =="B2":
                result_list[3] = round(label.classification.score*100, 1)
            elif label.display_name =="C1":
                result_list[4] = round(label.classification.score*100, 1)
            elif label.display_name =="C2":
                result_list[5] = round(label.classification.score*100, 1)
    return render_template('index.html', result=difficulty, text=form_data, result_list=result_list, given_sentence=given_sentence)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]