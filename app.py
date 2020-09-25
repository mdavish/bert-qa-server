import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify

model = BertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

import sys

sys.getsizeof(model)
sys.getsizeof(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device: {}'.format(device))

app = Flask(__name__)

def bert_qa(question, document):
    '''Takes a `question` string and an `document` string (which contains
    the answer), and identifies the words within the `document` that are
    the answer.
    '''
    input_ids = tokenizer.encode(question, document)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = model(torch.tensor([input_ids]),
                                          token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    confidence = float(max(torch.max(start_scores), torch.max(end_scores)))
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer, confidence

@app.route('/answer_question')
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


if __name__ == '__main__':
    app.run(debug=True)
