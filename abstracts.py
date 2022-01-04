from koala.encoders import Encoder


class Sample:
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label
        self.vector = None
        self.used_encoded = ""

    def encode(self, encoder: Encoder, keep_vector=False):
        if (self.vector is not None) and (self.used_encoded == encoder.model_to_use):
            return self.vector

        vector = encoder.encode(self.text)

        if keep_vector:
            self.vector = vector
            self.used_encoded = encoder.model_to_use
        return vector
