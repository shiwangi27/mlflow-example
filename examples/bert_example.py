import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] First, it describes the process by which the Committee came to recommend that the House impeach the President of the United States. [SEP] From start to finish, the House conducted its inquiry with a commitment to transparency, efficiency, and fairness. [SEP]"

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

masked_index = 17
tokenized_text[masked_index] = '[MASK]'

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)

sep_index = tokenized_text.index("[SEP]")

segment_ids = [0 if i <= sep_index else 1 for i in range(len(tokenized_text))]
print(segment_ids)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segment_ids])

print(tokens_tensor)
print(segments_tensor)

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensor)
    encoded_layers = outputs[0]
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
print(encoded_layers.shape)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensor)
    predictions = outputs[0]

print(predictions.shape)
print(predictions[0, masked_index])

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_index, predicted_token)





