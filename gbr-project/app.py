# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Load & train model
# data = pd.read_csv('Student_Performance.csv')
# le = LabelEncoder()
# if data['Extracurricular Activities'].dtype == 'object':
#     data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=10, test_size=0.2)

# gbr = GradientBoostingRegressor()
# gbr.fit(train_x, train_y)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     hr = int(request.form['hours'])
#     pre_score = int(request.form['score'])
#     activities = int(request.form['activities'])
#     sleep_hr = int(request.form['sleep'])
#     questions_practices = int(request.form['questions'])

#     res = gbr.predict([[hr, pre_score, activities, sleep_hr, questions_practices]])
#     return render_template('index.html', prediction=f"Predicted Performance: {res[0]:.2f}")
#     from flask import Flask, request, jsonify

#     @app.route('/predict_ajax', methods=['POST'])

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and train model
data = pd.read_csv('Student_Performance.csv')
le = LabelEncoder()
if data['Extracurricular Activities'].dtype == 'object':
    data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=10)

gbr = GradientBoostingRegressor()
gbr.fit(train_x, train_y)

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# AJAX prediction endpoint
@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    data = request.get_json()
    hr = int(data['hours'])
    pre_score = int(data['score'])
    activities = int(data['activities'])
    sleep_hr = int(data['sleep'])
    questions_practices = int(data['questions'])
    
    pred = gbr.predict([[hr, pre_score, activities, sleep_hr, questions_practices]])[0]
    return jsonify({'prediction': float(pred)})

if __name__ == '__main__':
    app.run(debug=True)
