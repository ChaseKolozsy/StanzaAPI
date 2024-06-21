from flask import Flask, request, jsonify
import stanza

app = Flask(__name__)

nlp = None
current_language = None

@app.route('/select_language', methods=['POST'])
def select_language():
    global nlp, current_language
    language = request.json.get('language')
    if not language:
        return jsonify({"error": "Language not provided"}), 400
    
    try:
        nlp = stanza.Pipeline(lang=language, processors='tokenize,pos,lemma,depparse')
        current_language = language
        return jsonify({"message": f"Language set to {language}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/process', methods=['POST'])
def process_text():
    if not nlp:
        return jsonify({"error": "Language not selected"}), 400
    
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "Text not provided"}), 400
    
    doc = nlp(text)
    result = []
    for sent in doc.sentences:
        sentence = {
            "text": sent.text,
            "tokens": [
                {
                    "text": word.text,
                    "lemma": word.lemma,
                    "pos": word.pos,
                    "deprel": word.deprel
                } for word in sent.words
            ]
        }
        result.append(sentence)
    
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)