import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score
from data_loader import get_validation_dataloader
from model_basic import Model, ModelConfig


def compute_eer(y_true, y_score):
    # Find the point where FPR and FNR are closest
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return ((fpr[idx]+fnr[idx])/2)*100, thresholds[idx]


@torch.no_grad()
def embed_sessions(model: Model, val_sequences, val_user_ids,device, batch_size):
    # returns tensor of shape (n_samples, emb_dim)
    loader = get_validation_dataloader(
        val_sequences, val_user_ids, batch_size=batch_size,
        sequence_length=model.config.seq_len,
        num_workers=0
    )
    embs = [] # (n_samples, d_model)
    for batch in loader:
        seqs = batch['sequences'].to(device)
        mm = batch['modality_mask'].to(device)
        emb, *_ = model(inputs=seqs, modality_mask=mm)# (B, d_model)
        embs.append(emb.detach().cpu()) 
    return torch.cat(embs, dim=0)

@torch.no_grad()
def score_user(enroll_emb, verify_emb, precision=None):
    # Cosine scores
    cos_mat = F.cosine_similarity(
        verify_emb.unsqueeze(1), enroll_emb.unsqueeze(0), dim=-1 # (n_samples, 1, d_model), (1, n_samples, d_model) => (n_samples_ver, n_samples_enr)
    ) # Cosine similarity between verify and enrollment embeddings
    cos_scores = cos_mat.mean(dim=1).cpu().numpy() # Averaging for each verification session (n_samples_ver)

    # Mahalanobis scores if precision provided
    if precision is not None:
        centered = verify_emb - enroll_emb.mean(0, keepdim=True) # Centering the verify_embedding
        # vectorized mahala: (x @ P * x).sum(dim=1)
        maha = -torch.einsum('bi,ij,bj->b', centered, precision, centered)
        maha_scores = maha.cpu().numpy()
    else:
        maha_scores = None

    return cos_scores, maha_scores

@torch.no_grad()
def validate(model, val_sequences, val_user_ids, val_user_to_indices: dict, device, batch_size=16):
    model.eval()
    cos_eers, mah_eers = [], []
    for u, sequence_indices in val_user_to_indices.items():
        
        user_sequences = [val_sequences[i] for i in sequence_indices]

        #(n_seqs, d_model)
        all_emb = embed_sessions(model=model, val_sequences=user_sequences, val_user_ids=[u] * len(user_sequences), device=device, batch_size=batch_size)

        split_point = len(all_emb) // 2
        # embeddings
        e_emb = all_emb[:split_point, :] # (n_samples, d_model)
        v_emb = all_emb[split_point:, :] # (n_samples, d_model)

        # precision for Mahalanobis
        centered = e_emb - e_emb.mean(0, True) # Centering the enrollment embeddings
        P = torch.tensor(
            LedoitWolf().fit(centered.cpu().numpy()).precision_.astype(np.float32)
        ) # Covariance matrix of the centered embeddings

        # genuine scores
        cos_g, mah_g = score_user(e_emb, v_emb, P)
        # impostor: pool all other users' first verify samples
        imp_emb = []

        other_user_sequences = []
        for other_user, other_user_sequence_indices in val_user_to_indices.items():
            if other_user == u: continue
            num_sequences = len(other_user_sequence_indices) // 2
            other_user_sequences.extend([val_sequences[i] for i in other_user_sequence_indices[:num_sequences]])

        imp_emb = embed_sessions(model=model, val_sequences=other_user_sequences, val_user_ids=[-1] * len(other_user_sequences),
                                  device=device, batch_size=batch_size)
        
        cos_i, mah_i = score_user(e_emb, imp_emb, P)

        # compute EERs
        eer_c, _ = compute_eer(
            np.concatenate([np.ones_like(cos_g), np.zeros_like(cos_i)]),
            np.concatenate([cos_g, cos_i])
        )
        cos_eers.append(eer_c)

        eer_m, _ = compute_eer(
            np.concatenate([np.ones_like(mah_g), np.zeros_like(mah_i)]),
            np.concatenate([mah_g, mah_i])
        )
        mah_eers.append(eer_m)

    model.train()
    return np.mean(cos_eers), np.mean(mah_eers)
