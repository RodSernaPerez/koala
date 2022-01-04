import transformers
from koala import config
import torch


class Encoder:
    def __init__(self, model_to_use: str):
        if model_to_use in config.DICT_MODELS.keys():
            model_to_use = config.DICT_MODELS[model_to_use]
        self.model = transformers.AutoModel.from_pretrained(model_to_use)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_to_use)
        self.model_to_use = model_to_use

    @classmethod
    def pooling_function(cls, tensor_output):
        return torch.mean(torch.squeeze(tensor_output), axis=0).detach().numpy()

    def encode(self, text: str):
        tokenized = self.tokenizer(text, return_tensors='pt', truncation=True)
        output = self.model(**tokenized)[0]
        flatten_output = self.pooling_function(output)
        return flatten_output
