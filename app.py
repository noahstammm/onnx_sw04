import onnxruntime
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image


app = Flask(__name__)


sess = onnxruntime.InferenceSession('./model/SSD.onnx')
input_name = sess.get_inputs()[0].name


@app.route('/')
def home():

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print('hi')
    file = request.files['image']
    print("test")
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(file)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # Ausführen der Inferenz auf dem ONNX-Modell
    outputs = sess.run(None, {input_name: img})

    # Formatieren der Ausgabe als JSON
    result = []
    for i in range(outputs[0].shape[1]):
        label = int(outputs[0][0, i, 1])
        confidence = float(outputs[0][0, i, 2])
        if confidence > 0.5:
            result.append({
                'label': label,
                'confidence': confidence
            })

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
