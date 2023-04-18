from flask import Flask, render_template, request
import onnxruntime
import numpy as np
import cv2
import torch

app = Flask(__name__)

# Laden des ResNet50-Modells
sess = onnxruntime.InferenceSession('model/resnet101-v2-7.onnx')

# Definition der Eingabegröße des Modells
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_dtype = sess.get_inputs()[0].type


# Laden der Klassenbezeichnungen
with open('model/synset.txt', 'r') as f:
    labels = [line.strip() for line in f]


@app.route('/')
def index():
    print(input_dtype)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Laden des Bildes aus dem HTML-Formular
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Skalierung des Bildes auf die Größe, das das Modell erwartet
    img_resized = cv2.resize(img, tuple(input_shape[2:]))



    # Konvertierung des Bildes in das Format, das das Modell erwartet
    input_dtype = np.float32
    img_np = np.asarray(img_resized, dtype=input_dtype)

    #img_np = torch.tensor(img_resized, dtype=torch.float)
    img_np = img_np.transpose(2, 0, 1)
    img_np = img_np.reshape(input_shape)

    # Generierung der Vorhersage
    outputs = sess.run(None, {input_name: img_np})
    predictions = np.squeeze(outputs[0])

    # Anzeigen der Top-5-Klassenbezeichnungen mit den höchsten Vorhersagewerten
    top_indexes = np.argsort(predictions)[::-1][:5]
    top_labels = [labels[i] for i in top_indexes]
    top_scores = [predictions[i] for i in top_indexes]

    # Rückgabe der Vorhersage als JSON-Objekt an das HTML-Formular
    return {'labels': top_labels, 'scores': top_scores}


if __name__ == '__main__':
    app.run(debug=True)
