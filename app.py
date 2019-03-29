from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

import pickle
import pandas as pd
import os
import numpy as np


# Preparing the Classifier
cur_dir = os.path.dirname('__file__')
clf = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelsales.pkl'), 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
	
	return render_template('index2.html')

@app.route('/results', methods=['POST'])
def predict():
	features1 = int(request.form['date'])
	features2 = int(request.form['id'])
	

	input_data = [{'date': features1, 'id': features2 }]
	data = pd.DataFrame(input_data)
	logreg = clf.predict(data)[0]
	return render_template('results.html', res=logreg)

if __name__ == '__main__':
	app.run(debug=True)