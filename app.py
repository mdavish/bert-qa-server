from bert_qa import BERTQA
from dpr_reader import DPRReader
from utils import boilerpipe_from_url
from flask import Flask, request, jsonify


app = Flask(__name__)
bertqa = BERTQA()
dpr_reader = DPRReader()
device = dpr_reader.device
print(f'Using device: {device}')


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
    #TODO Clean this up, make it only accept GET requests.
    payload = request.json
    question = payload['question']
    url = payload['url']
    method = payload['method']
    bp_response = boilerpipe_from_url(url)
    bp_content = bp_response['response']['content']
    if method == 'squad':
        response = bertqa.answer_question_chunked(question, bp_content)
    elif method == 'dpr':
        response = dpr_reader.read_chunked_document(question, bp_content, '')
    else:
        raise ValueError(f'Invalid extaction method method - {method}.')
    return jsonify(response)
