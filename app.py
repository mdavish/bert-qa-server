import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from flask import Flask, request, jsonify

tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased', return_token_type_ids=True)
model = DistilBertForQuestionAnswering.from_pretrained(
    'distilbert-base-uncased-distilled-squad')

app = Flask(__name__)

def bert_qa(question, document):
    '''Takes a `question` string and an `document` string (which contains
    the answer), and identifies the words within the `document` that are
    the answer.
    '''
    encoding = tokenizer.encode_plus(question, document)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]),
                                      attention_mask=torch.tensor([attention_mask]))
    confidence = float(max(torch.max(start_scores), torch.max(end_scores)))
    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens,
                                                    skip_special_tokens=True)
    answer = answer_tokens[0]
    for token in answer_tokens[1:]:
        if token[0:2] == '##':
            answer += token[2:]
        else:
            answer += ' ' + token
    return answer, confidence

@app.route('/')
def answer_question():
    payload = request.json
    question = payload['question']
    documents = payload['documents']
    full_response = []
    for document in documents:
        answer, confidence = bert_qa(question, document)
        confidence = float(confidence)
        response = {
            'document': document,
            'answer': answer,
            'confidence': confidence
        }
        full_response.append(response)
    return jsonify(full_response)
