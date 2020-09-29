import re
import logging
import requests
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


class BERTQA:

    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased', return_token_type_ids=True)
    model = DistilBertForQuestionAnswering.from_pretrained(
        'distilbert-base-uncased-distilled-squad')
    MAX_TOKENS = 512
    MAX_TOKENS_QUESTION = 30
    MAX_TOKENS_DOCUMENT = MAX_TOKENS - MAX_TOKENS_QUESTION - 2  # [SEP] and [CLS]

    def __init__(self):
        pass

    def get_token_length(self, string):
        tokens = self.tokenizer.encode(string)
        return len(tokens)

    def chunk_document(self, document, re_consolidate=True):
        '''Chunks up a long document into optimally large pieces so that they
        can be passed to BERT. Activating `re_consolidate` will put the chunks
        back together to make them as large as possible for improved
        performance.
        '''
        document_length = self.get_token_length(document)
        if document_length > self.MAX_TOKENS_DOCUMENT:
            approved_chunks = []
            paragraphs = document.split('\n')
            paragraphs = [par for par in paragraphs if par]
            for paragraph in paragraphs:
                paragraph_length = self.get_token_length(paragraph)
                if paragraph_length > self.MAX_TOKENS_DOCUMENT:
                    sentences = paragraph.split('.')
                    sentences = [sen for sen in sentences if sen]
                    for sentence in sentences:
                        sentence_length = self.get_token_length(sentence)
                        if sentence_length > self.MAX_TOKENS_DOCUMENT:
                            print("Ignoring overlong sentence.")
                        else:
                            approved_chunks.append(sentence)
                else:
                    approved_chunks.append(paragraph)
            if re_consolidate:
                lengths = [self.get_token_length(
                    chunk) for chunk in approved_chunks]
                consolidated_chunks = []
                running_length = 0
                current_chunk = ''
                for chunk, length in zip(approved_chunks, lengths):
                    if (running_length + length) < self.MAX_TOKENS_DOCUMENT:
                        current_chunk += chunk
                        running_length += length
                    else:
                        consolidated_chunks.append(current_chunk)
                        current_chunk = chunk
                        running_length = length
                return consolidated_chunks
            else:
                return approved_chunks
        else:
            return [document]

    def answer_question(self, question, document):
        '''Takes a `question` string and an `document` string (which contains
        the answer), and identifies the words within the `document` that are
        the answer.
        '''
        question_length = self.get_token_length(question)
        document_length = self.get_token_length(document)
        if question_length > self.MAX_TOKENS_QUESTION:
            msg = f'Question exceeds max token length ({str(question_length)}).'
            raise ValueError(msg)
        if document_length > self.MAX_TOKENS_DOCUMENT:
            msg = f'Document exceeds max token length ({str(document_length)}).'
            raise ValueError(msg)
        encoding = self.tokenizer.encode_plus(question, document)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        start_scores, end_scores = self.model(torch.tensor([input_ids]),
                                              attention_mask=torch.tensor([attention_mask]))
        confidence = float(max(torch.max(start_scores), torch.max(end_scores)))

        start_token = torch.argmax(start_scores)
        end_token = torch.argmax(end_scores)
        ans_tokens = input_ids[torch.argmax(
            start_scores): torch.argmax(end_scores) + 1]
        answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens,
                                                             skip_special_tokens=True)
        if not answer_tokens:  # TODO Understand this bug
            return '<NO ANSWER>', -10
        else:
            answer = answer_tokens[0]
            for token in answer_tokens[1:]:
                if token[0:2] == '##':
                    answer += token[2:]
                else:
                    answer += ' ' + token
            return answer, confidence

    def answer_question_chunked(self, question, document, re_consolidate=True):
        chunks = self.chunk_document(document, re_consolidate=True)
        responses = []
        for chunk in tqdm(chunks):
            answer, confidence = self.answer_question(question, chunk)
            response = {
                'answer': answer,
                'confidence': confidence,
                'chunk': chunk
            }
            responses.append(response)
        responses.sort(key=lambda x: -x['confidence'])
        return responses
