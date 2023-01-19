from flask import Flask,render_template,request,session,url_for,jsonify,flash,send_from_directory
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
img_size=100
app = Flask(__name__)
app.secret_key='Lakshmi'



@app.route('/')
def index():
    return render_template("home.html")


@app.route('/ct')
def ct():
    return render_template("index.html")


@app.route("/upload", methods=["POST","GET"])
def upload():
    print('a')
    if request.method=='POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('images/', fn)
        myfile.save(mypath)
        print(fn)
        print(type(fn))
        accepted_formated=['jpg','png','jpeg','jfif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted","Danger")
        # mypath="dataset/train/COVID NEGATIVE/covid_negative_1_2185.png"
        new_model = load_model("alg/Resnet50.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        print(result)
        print(np.argmax(result))
        accuracy = float(np.max(result, axis=1)[0])
        accuracy = accuracy*100
        a = "Accuracy of Image: " + str(accuracy) +str('%')

        classes=['Covid-19 Negative', 'Covid-19 Positive']
        prediction=classes[np.argmax(result)]
        if prediction=="Covid-19 Positive":
            msg="Covid-19 precautions:: Preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. The use of face masks or coverings has been recommended in public settings to minimise the risk of transmissions."
        else:
            msg="You didnt detect any covid virus in you lungs but please wear masks all the time and wash your hands frequently"

        return render_template("template.html",image_name=fn, text=prediction,msg=msg, a= a)
    return render_template("template.html")

@app.route("/upload1", methods=["POST","GET"])
def upload1():
    print('a')
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)
        print(fn)
        print(type(fn))
        accepted_formated = ['jpg', 'png', 'jpeg', 'jfif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted", "Danger")
        mypath="covid-CT Scan/dataset/train/COVID POSITIVE/covid_positive_1_2310.png"
        new_model = load_model(r"alg/FinalModel.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        print(result)
        print(np.argmax(result))
        result = np.argmax(result, axis=1)[0]
        accuracy = float(np.max(result, axis=1)[0])
        classes = ['Covid-19 Negative', 'Covid-19 Positive']
        print(classes)
        a = "Accuracy of Image: " + str(accuracy)
        prediction = classes[np.argmax(result)]
        print(prediction)
        if prediction == 'Covid-19 Positive':
            mag1 = "Covid-19 precautions:: Preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. The use of face masks or coverings has been recommended in public settings to minimise the risk of transmissions."
        else:
            mag1 = ""
    return render_template("template.html", image_name=fn, text=prediction, msg=mag1, a=accuracy)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/classify')
def classify():
    return render_template("classify.html")

if __name__ == '__main__':
    app.run(debug=True,port=9000)