from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)
model = joblib.load('model.pkl')


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/performance")
def performance():
    return render_template("performance.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/eda1")
def eda1():
    return render_template("eda1.html")

@app.route("/eda2")
def eda2():
    return render_template("eda2.html")

@app.route("/eda3")
def eda3():
    return render_template("eda3.html")

@app.route("/eda4")
def eda4():
    return render_template("eda4.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]

    if int_features[0]==0:
        f_features=[0,0,0]+int_features[1:]
    elif int_features[0]==1:
        f_features=[1,0,0]+int_features[1:]
    elif int_features[0]==2:
        f_features=[0,1,0]+int_features[1:]
    else:
        f_features=[0,0,1]+int_features[1:]

    if f_features[6]==0:
        fn_features=f_features[:6]+[0,0]+f_features[7:]
    elif f_features[6]==1:
        fn_features=f_features[:6]+[1,0]+f_features[7:]
    else:
        fn_features=f_features[:6]+[0,1]+f_features[7:]

    final_features = [np.array(fn_features)]
    predict = model.predict(final_features)

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return render_template('prediction.html', output=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    predict = model.predict([np.array(list(data.values()))])

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return jsonify(output)


if __name__ == "__main__":
    app.debug = False
    app.run()
