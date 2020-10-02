from typing import List
import torch
from transformers import DPRReader, DPRReaderTokenizer


class DPRReader:

    reader_tokenizer = DPRReaderTokenizer.from_pretrained(
        'facebook/dpr-reader-single-nq-base')
    reader_model = DPRReader.from_pretrained(
        'facebook/dpr-reader-single-nq-base', return_dict=True)
    MAX_TOKENS = 512
    MAX_TOKENS_QUESTION = 30
    MAX_TOKENS_DOCUMENT = MAX_TOKENS - MAX_TOKENS_QUESTION - 2  # [SEP] and [CLS]

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.reader_model = self.reader_model.cuda()

    def _reconstruct_tokens(self, bert_tokens):
        output_string = ''
        for token in bert_tokens:
            if token[:2] == '##':
                output_string += token[2:]
            else:
                output_string += ' '
                output_string += token
        return output_string[1:]


    def get_token_length(self, string):
        tokens = self.reader_tokenizer.encode(string)
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


    def read_documents(self, question: str, documents: List[str], titles: List[str]):
        encoded_inputs = self.reader_tokenizer(
            questions=question,
            titles=titles,
            texts=documents,
            return_tensors='pt',
            padding=True
        )
        input_ids = encoded_inputs['input_ids']
        encoded_inputs = encoded_inputs.to(self.device) #TODO Figure this out?
        outputs = self.reader_model(**encoded_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        relevance_logits = outputs.relevance_logits
        responses = []
        for i in range(len(documents)):
            title = titles[i]
            document = documents[i]
            start = start_logits[i]
            end = end_logits[i]
            relevance = relevance_logits[i]
            inp_ids = input_ids[i]
            input_tokens = self.reader_tokenizer.convert_ids_to_tokens(inp_ids)
            answer_start = int(start.argmax())
            answer_end = int(end.argmax())
            relevance = float(relevance.max())
            answer_tokens = input_tokens[answer_start : answer_end + 1]
            answer_str = self._reconstruct_tokens(answer_tokens)
            response = {
                'answer': answer_str,
                'relevance': relevance,
                'title': title,
                'document': document
            }
            responses.append(response)
        response = responses.sort(key=lambda x: -x['relevance'])
        return responses

    def read_chunked_document(self, question: str, document: str, title: str):
        chunked_docs = self.chunk_document(document)
        titles_list = [title for i in range(len(chunked_docs))]
        return self.read_documents(question, chunked_docs, titles_list)
