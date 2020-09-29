from bert_qa import BERTQA
from utils import boilerpipe_from_url
from flask import Flask, request, jsonify


app = Flask(__name__)
bertqa = BERTQA()


@app.route('/qa_from_documents', methods=['GET','POST'])
def qa_from_documents():
    payload = request.json
    question = payload['question']
    documents = payload['documents']
    full_response = []
    for document in documents:
        answer, confidence = bertqa.answer_question(question, document)
        confidence = float(confidence)
        response = {
            'document': document,
            'answer': answer,
            'confidence': confidence
        }
        full_response.append(response)
    return jsonify(full_response)


@app.route('/qa_from_url', methods=['GET','POST'])
def qa_from_url():
    payload = request.json
    question = payload['question']
    url = payload['url']
    bp_response = boilerpipe_from_url(url)
    bp_content = bp_response['response']['content']
    response = bertqa.answer_question_chunked(question, bp_content)
    return jsonify(response)
