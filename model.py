import torch.nn as nn 
import torch
class QAScorer(nn.Module) :
    def __init__(self,emb_dim=100,hidden_dim=128) :  # Embedding Dimension = Input vector ki length.
        super(QAScorer, self).__init__()

        input_dim = emb_dim * 3   # promt_vec + option_vec + |diff| 

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(hidden_dim // 2 , 1)                # ek score output
        )

    def forward(self, prompt_vec, option_vec) :
        diff = torch.abs(prompt_vec - option_vec)
        x = torch.cat([prompt_vec, option_vec, diff], dim=1)
        score = self.network(x)      # shape: (batch, 1)
        return score.squeeze(-1)    # shape: (batch,)

class MultipleChoiceModel(nn.Module):
    def __init__(self, emb_dim=100, hidden_dim=128, num_options=5): 
        super(MultipleChoiceModel, self).__init__() 
        self.scorer = QAScorer(emb_dim, hidden_dim)
        self.num_options = num_options 

    def forward(self, prompt_vec, option_vecs):
        # prompt_vec: (batch, emb_dim)
        # option_vecs: (batch, num_options, emb_dim)
        scores = []
        for i in range(self.num_options) :
            s = self.scorer(prompt_vec, option_vecs[:,i,:])
            scores.append(s) 
        logits = torch.stack(scores, dim=1)           # (batch, num_options)
         
        return logits