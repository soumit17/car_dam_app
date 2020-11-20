from flask import Flask, request
import numpy as np
from flasgger import Swagger
import tensorflow.keras
from PIL import Image, ImageOps
import io
import base64

app=Flask(__name__)
Swagger(app)

np.set_printoptions(suppress=True)
# model_cardam = tensorflow.keras.models.load_model('/home/ubuntu/car_dam_app/damage_whole_model/keras_model.h5')
model_state = tensorflow.keras.models.load_model('/home/ubuntu/mlapp/damage_whole_model/keras_model.h5')
model_geo = tensorflow.keras.models.load_model('/home/ubuntu/mlapp/front_rear_side_model/keras_model.h5')
model_severity = tensorflow.keras.models.load_model('/home/ubuntu/mlapp/minor_moderate_severe_model/keras_model.h5')

# model_state = tensorflow.keras.models.load_model('damage_whole_model/keras_model.h5')
# model_geo = tensorflow.keras.models.load_model('front_rear_side_model/keras_model.h5')
# model_severity = tensorflow.keras.models.load_model('minor_moderate_severe_model/keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

state_labels=['damage', 'whole']
geo_labels= ['front', 'rear', 'side']
severity_labels=['minor', 'moderate', 'severe']

def image_pred(im):
    size = (224, 224)
    image = ImageOps.fit(im, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    pred_state = model_state.predict(data)
    pred_geo = model_geo(data)
    pred_severity = model_severity(data)
    
    state = state_labels[list(pred_state[0]).index(max(pred_state[0]))]
    geo = geo_labels[list(pred_geo[0]).index(max(pred_geo[0]))]
    severity = severity_labels[list(pred_severity[0]).index(max(pred_severity[0]))]
    
    if state == 'damage':
        result = 'Car has ' + severity + ' damage on ' + geo
    else:
        result ='Car is good on '+ geo
    return result

@app.route('/car_damage_B64',methods=["POST"])
def predict_car_damage_B64():
    """Check car is damaged or not
    This is using docstrings for specifications.
    ---
    parameters:
      - name: image
        in: formData
        type: text
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    inputdata = request.form['image']
    imgdata=base64.b64decode(bytes(str(inputdata)[2:], 'ascii'))
    im = Image.open(io.BytesIO(imgdata))
    return image_pred(im)

@app.route('/car_damage',methods=["POST"])
def predict_car_damage():
    """Check car is damaged or not
    This is using docstrings for specifications.
    ---
    parameters:
      - name: imagefile
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    im = Image.open(request.files.get("imagefile"))
    return image_pred(im)



