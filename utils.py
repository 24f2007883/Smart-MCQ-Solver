import string
import numpy as np
import string

EMB_DIM = 100 
def sentence_vector(text, model, dim=EMB_DIM):

    tokens = clean(text).split()

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)

def clean(text) :
    text = text.lower() 
    text = text.translate(
        str.maketrans('','', string.punctuation)
    )
    return text 