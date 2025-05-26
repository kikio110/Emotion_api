from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from deep_translator import GoogleTranslator
import json
from pre import *

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_lstm.tflite")
interpreter.allocate_tensors()

# Ambil detail input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load tokenizer
with open("tokenizer.json") as f:
    data = f.read()  # ambil sebagai string, bukan json.load()
    tokenizer = tokenizer_from_json(data)


# Label kategori (sesuaikan dengan modelmu)
label = ["sadness", "joy", "love", "anger", "fear", "surprise"]

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Emotion Model API using TFLite!',
        'status': 'success'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'kalimat' not in data:
            return jsonify({'error': 'Parameter "kalimat" dibutuhkan'}), 400

        kalimat_asli = data['kalimat']

        # Translate ke Bahasa Inggris
        kalimat_terjemahan = GoogleTranslator(source='auto', target='en').translate(kalimat_asli)

        # Preprocessing
        kalimat_clean = cleaningText(kalimat_terjemahan)
        kalimat_casefolded = casefoldingText(kalimat_clean)
        kalimat_slangfixed = fix_slangwords(kalimat_casefolded)
        kalimat_tokenized = tokenizingText(kalimat_slangfixed)
        kalimat_filtered = filteringText(kalimat_tokenized)
        kalimat_lemmatized = lemmatizationText(kalimat_filtered)
        kalimat_final = toSentence(kalimat_lemmatized)

        # Tokenizing dan padding
        sequence = tokenizer.texts_to_sequences([kalimat_final])
        input_data = pad_sequences(sequence, maxlen=200, padding='post')

        # Prediksi dengan model TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = int(np.argmax(output_data))
        emosi = label[predicted_index]

        return jsonify({
            'label_prediksi': predicted_index,
            'emosi': emosi
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
