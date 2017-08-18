from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
import numpy as np
import os
import base64
import tensorflow as tf, sys
import PIL
from PIL import Image

basewidth = 150
sess = tf.Session()

with tf.gfile.FastGFile(PATH_TO_MODEL_HERE, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

label_lines = [line.rstrip() for line in tf.gfile.GFile(PATH_TO_LABELS_HERE)]
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def getresults(image_data):
    print("We are here")
    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    i = 0
    print("received predictions")
    rc = []
    rs = []
    for node_id in top_k:
        rc.append(label_lines[node_id])
        rs.append(predictions[0][node_id])
        print('%s (score = %.5f)' % (rc[i], rs[i]))
        i = i+1
        if i==5:
            break
        print("Predictions stored")
    data = {
        'rc1': rc[0],
        'rs1': str(rs[0]),
        'rc2': rc[1],
        'rs2': str(rs[1]),
        'rc3': rc[2],
        'rs3': str(rs[2]),
        'rc4': rc[3],
        'rs4': str(rs[3]),
        'rc5': rc[4],
        'rs5': str(rs[4]),
        }
    print("Predictions send")
    return jsonify(data)

app = Flask(__name__)
@app.route('/data', methods=['GET','POST'])
def data_page():
    print("Reached here")
    image_b64 = request.values['imageBase64']
    print(len(image_b64))
    content = image_b64.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(image_encoded.encode('utf-8'))
    with open("image.jpg", "wb") as fh:
        fh.write(body)
    img = Image.open('image.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    img.save('imagenew.jpg')
    return 'OK'

@app.route('/classify', methods=['GET','POST'])
def classify():
    image_path = 'PATH_TO_SAVE_IMAGE/imagenew.jpg'
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    result = getresults(image_data)
    return result

@app.route('/')
def home():
    return render_template('UserInterface.html')

if __name__=='__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True, port=9000, host='0.0.0.0')
