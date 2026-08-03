import torch
import gradio as gr
from gensim.models import Word2Vec

from model import MultipleChoiceModel
from utils import sentence_vector
import numpy as np

EMB_DIM = 100

w2v_model = Word2Vec.load("word2vec.model")

model = MultipleChoiceModel(
    emb_dim=EMB_DIM,
    hidden_dim=128
)

model.load_state_dict(
    torch.load("best_model.pth", map_location="cpu")
)

model.to("cpu")
model.eval()

import torch.nn.functional as F

idx_to_option = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

def predict(prompt, A, B, C, D, E):

    # Prompt vector
    prompt_vec = sentence_vector(prompt, w2v_model)

    # Option vectors
    option_vecs = [
        sentence_vector(A, w2v_model),
        sentence_vector(B, w2v_model),
        sentence_vector(C, w2v_model),
        sentence_vector(D, w2v_model),
        sentence_vector(E, w2v_model)
    ]

    # Convert to tensor
    prompt_tensor = torch.tensor(prompt_vec, dtype=torch.float32).unsqueeze(0)

    option_tensor = torch.tensor(
        np.array(option_vecs),
        dtype=torch.float32
    ).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        logits = model(prompt_tensor, option_tensor)
        print(logits)

        probs = F.softmax(logits, dim=1)

    probs = probs.squeeze(0)

    top3 = torch.topk(probs, k=3)

    prediction = idx_to_option[top3.indices[0].item()]

    result = ""
    print(prompt_vec.shape)

    print(option_vecs[0][:5])
    print(option_vecs[1][:5])
    print(option_vecs[2][:5])
    print(option_vecs[3][:5])
    print(option_vecs[4][:5])

    for idx, score in zip(top3.indices, top3.values):

        result += f"Option {idx_to_option[idx.item()]} : {score.item()*100:.2f}%\n"

    return prediction, result

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=3, label="Question / Prompt"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Answer"),
        gr.Textbox(label="Top 3 Predictions"),
    ],
    title="Smart MCQ Solver",
    description="Word2Vec + PyTorch Scratch Model"
).launch()