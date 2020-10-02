# BERT Q&A

This repository contains some utilities including a REST endpoint for doing
question-answering with BERT! There are classes for both SQUAD BERT (`BERTQA`)
and for DPR BERT (`DPRReader`).

## Using the Flask App Endpoint
```
gunicorn app:app
curl --location --request GET 'localhost:5001/' \
--header 'Content-Type: application/json' \
--data-raw '{
    "question": "Who went to UMichigan?",
    "documents": ["Marc Ferrentino went to UMichigan. Howard Lerman went to Thomas Jefferson High School."]
}'
```

## Using the DPR Class
```python
from dpr_reader import DPRReader
question = 'Who is LeBron James?'
documents = [
    'Lebron James is a basketball player.',
    'Jeff Bezos is an American CEO.',
    'Cleopatra was an Egyptian Pharoah.'
]
titles = [
    'All About LeBron James',
    'All About Jeff Bezos',
    'All About Cleopatra'
]

dpr_reader = DPRReader()
outputs = dpr_reader.read_documents(question, documents, titles)
```

## Using the BERTQA Class
```python
from bert_qa import BERTQA
question = 'Who is LeBron James?'
documents = [
    'Lebron James is a basketball player.',
    'Jeff Bezos is an American CEO.',
    'Cleopatra was an Egyptian Pharoah.'
]

bert_qa = BERTQA()
outputs = bert.answer_question(question, documents)
```
