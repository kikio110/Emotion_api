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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'kalimat' not in data:
        return jsonify({'error': 'Parameter "kalimat" dibutuhkan'}), 400

    kalimat_asli = data['kalimat']

    # Translate ke Bahasa Indonesia
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
