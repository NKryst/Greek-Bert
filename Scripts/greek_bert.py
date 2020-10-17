from transformers import AutoTokenizer, AutoModelForPreTraining

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

model = AutoModelForPreTraining.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")