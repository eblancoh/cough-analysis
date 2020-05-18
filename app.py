from flask import Flask
from flask import request
from flask import render_template, jsonify
import os, sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    if request.method == "POST":
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            f = open('./file.wav', 'wb')
            f.write(request.get_data("audio_data"))
            f.close()
            # Procesamos el dato de audio facilitado al frontal
            y, sr = librosa.load(request.files['audio_data'])

            # Let's make and display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

            # Convert to log scale (dB). We'll use the peak power as reference.
            # log_S = librosa.logamplitude(S, ref_power=np.max)
            log_S = librosa.amplitude_to_db(S, ref=np.max)


            # Make a new figure
            fig = plt.figure(figsize=(12,4))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # Display the spectrogram on a mel scale
            # sample rate and hop length parameters are used to render the time axis
            librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

            # Make the figure layout compact

            image_path = 'testing/tmp.png'
            #plt.show()
            plt.savefig(image_path)
            plt.close()

            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            # Loads label file, strips off carriage return
            label_lines = [line.rstrip() for line
                            in tf.gfile.GFile("retrained_labels.txt")]

            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            payload = {
                "diagnosis": label_lines[top_k[0]], \
                "score": predictions[0][top_k[0]]
            }
            
            print(payload)
            # print(request.files['audio_data'])
            if os.path.isfile('./file.wav'):
                print("file.wav exists")

        return render_template('index.html', request="POST")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, \
            threaded=True, \
            port="5000")