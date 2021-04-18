import unicodedata

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

accented_string = "Αυτή είναι η Ελληνική έκδοση του BERT."
unaccented_string = strip_accents_and_lowercase(accented_string)

print(unaccented_string) # αυτη ειναι η ελληνικη εκδοση του bert.

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

import torch
from transformers import *

# Load model and tokenizer
tokenizer_greek = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
lm_model_greek = AutoModelWithLMHead.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

# ================ EXAMPLE 1 ================
text_1 = 'O ποιητής έγραψε ένα [MASK] .'
# EN: 'The poet wrote a [MASK].'
input_ids = tokenizer_greek.encode(text_1)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'o', 'ποιητης', 'εγραψε', 'ενα', '[MASK]', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 5].max(0)[1].item()))
# the most plausible prediction for [MASK] is "song"

# ================ EXAMPLE 2 ================
text_2 = 'Είναι ένας [MASK] άνθρωπος.'
# EN: 'He is a [MASK] person.'
input_ids = tokenizer_greek.encode(text_1)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 3].max(0)[1].item()))
# the most plausible prediction for [MASK] is "good"

# ================ EXAMPLE 3 ================
text_3 = 'Είναι ένας [MASK] άνθρωπος και κάνει συχνά [MASK].'
# EN: 'He is a [MASK] person he does frequently [MASK].'
input_ids = tokenizer_greek.encode(text_3)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', 'και', 'κανει', 'συχνα', '[MASK]', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 8].max(0)[1].item()))
# the most plausible prediction for the second [MASK] is "trips"
