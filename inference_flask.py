import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from flask import Flask, jsonify, render_template, request
app = Flask(__name__, template_folder='html')  

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Dense(1))

model = tf.keras.models.load_model('prof.h5', custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)

@app.route('/')  
def upload():  
    return render_template("upload.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.form['text']  
        t = model.predict([f[0]])
        t = t.tolist()
        # return jsonify(amul = t[0])
        return render_template("success.html", toxicity = t)  

if __name__ == '__main__':  
    app.run(debug = False)  

# while True:
#     print(model.predict([x]))