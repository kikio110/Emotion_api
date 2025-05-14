from flask import Flask, request, jsonify
import joblib
from deep_translator import GoogleTranslator
from pre import *

# Load model dan vectorizer
naive_bayes = joblib.load('naive_bayes_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Label emosi
label = ["sadness", "joy", "love", "anger", "fear", "surprise"]

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Emotion Model API!',
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

        # TF-IDF dan prediksi
        X_input = tfidf.transform([kalimat_final])
        prediksi = naive_bayes.predict(X_input.toarray())[0]
        emosi = label[prediksi]

        return jsonify({
            'label_prediksi': int(prediksi),
            'emosi': emosi
        })

    except Exception as e:
        # Print or log the error so you can see it
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
