import numpy as np
import torch
import torch.nn.functional as F
import math
import itertools
import matplotlib.pyplot as plt
import time
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data_loader import get_validation_dataloader, get_testing_dataloader
from model_basic import Model, ModelConfig
from matplotlib.patches import Patch, Ellipse


def compute_eer(y_true, y_score, plot=False):
    # Find the point where FPR and FNR are closest
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = ((fpr[idx]+fnr[idx])/2)*100
    if plot: 
        plt.figure(figsize=(6, 6))
        plt.plot(fpr * 100, fnr * 100, lw=2)
        plt.xlabel("False Acceptance Rate (%)")
        plt.ylabel("False Rejection Rate (%)")
        plt.title("")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.plot([0, 100], [0, 100], "--", color="gray", alpha=0.7)  # reference diagonal
        plt.tight_layout()
        plt.savefig(f"./exp3/res/{eer}.png")

    auc = roc_auc_score(y_true, y_score) # Area under the curve

    return eer, thresholds[idx], auc


@torch.no_grad()
def embed_sessions(model: Model, val_sequences, val_user_ids,device, batch_size):
    # returns tensor of shape (n_samples, emb_dim)
    loader = get_testing_dataloader(
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
def score_user(enroll_emb, verify_emb, precision=None, get_cosine_score=True):
    cos_scores = None

    if get_cosine_score: 
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
def validate(model, val_sequences, val_user_ids, val_user_to_indices: dict, device, batch_size=16, all_imp = True, plot=False, get_cosine_score=True, num_of_enroll_seqs=None):
    model.eval()
    cos_eers, mah_eers = [], []
    cos_aucs, mah_aucs = [], []
    all_user_embs = []
    all_user_labels = []
    for u, sequence_indices in val_user_to_indices.items():
        
        user_sequences = [val_sequences[i] for i in sequence_indices]

        #(n_seqs, d_model)
        all_emb = embed_sessions(model=model, val_sequences=user_sequences, val_user_ids=[u] * len(user_sequences), device=device, batch_size=batch_size)

        split_point = num_of_enroll_seqs if (num_of_enroll_seqs is not None and len(all_emb) > num_of_enroll_seqs) else len(all_emb) // 2
        # embeddings
        e_emb = all_emb[:split_point, :] # (n_samples, d_model)
        v_emb = all_emb[split_point:, :] # (n_samples, d_model)
        if plot:
            all_user_embs.append(v_emb.numpy())
            all_user_labels.extend([u] * v_emb.size(0))

        # precision for Mahalanobis
        centered = e_emb - e_emb.mean(0, True) # Centering the enrollment embeddings
        P = torch.tensor(
            LedoitWolf().fit(centered.cpu().numpy()).precision_.astype(np.float32)
        ) # Covariance matrix of the centered embeddings

        # genuine scores
        cos_g, mah_g = score_user(e_emb, v_emb, P, get_cosine_score=get_cosine_score)
        # impostor: pool all other users' first verify samples
        imp_emb = []

        other_user_sequences = []        
        sequence_to_take_per_user = math.ceil(len(v_emb) / (len(val_user_to_indices.keys()) - 1))
        # print("Sequence To Take Per User", sequence_to_take_per_user)
        for other_user, other_user_sequence_indices in val_user_to_indices.items():
            if other_user == u: continue

            num_sequences = len(other_user_sequence_indices) // 2 if all_imp else sequence_to_take_per_user
            other_user_sequences.extend([val_sequences[i] for i in other_user_sequence_indices[:num_sequences]])
            if len(other_user_sequences) >= len(v_emb) and not all_imp:
                break

        imp_emb = embed_sessions(model=model, val_sequences=other_user_sequences, val_user_ids=[-1] * len(other_user_sequences),
                                  device=device, batch_size=batch_size)
        
        print(f"User: {u} | # enrollment: {len(e_emb)} | # verify: {len(v_emb)} | # imposter: {len(imp_emb)}")

        cos_i, mah_i = score_user(e_emb, imp_emb, P, get_cosine_score=get_cosine_score)

        # compute EERs
        eer_c, auc_c = None, None
        if get_cosine_score:
            eer_c, _, auc_c = compute_eer(
                np.concatenate([np.ones_like(cos_g), np.zeros_like(cos_i)]),
                np.concatenate([cos_g, cos_i]), plot=plot
            )
            cos_eers.append(eer_c)
            cos_aucs.append(auc_c)

        eer_m, _, auc_m = compute_eer(
            np.concatenate([np.ones_like(mah_g), np.zeros_like(mah_i)]),
            np.concatenate([mah_g, mah_i]), plot=plot
        )
        mah_eers.append(eer_m)
        mah_aucs.append(auc_m)

        print(f"User: {u} | EERs: | cosine: {eer_c}  | maha: {eer_m} | cosine_auc: {auc_c} | maha_auc: {auc_m}")
    model.train()

    if plot: 
        all_user_embs = np.vstack(all_user_embs)
        all_user_labels = np.array(all_user_labels, dtype=int)
        plot_tsne_with_ellipses(embeddings=all_user_embs, labels=all_user_labels, title="")

    return np.mean(cos_eers), np.mean(mah_eers), np.mean(cos_aucs), np.mean(mah_aucs)



@torch.no_grad()
def validate_multi(
    model: Model,
    val_sequences,
    val_user_ids,
    val_user_to_indices: dict,
    device,
    batch_size=16,
    group_sizes=(2, 3, 4, 5),
):
    """
    Multi‐user Mahalanobis validation with group‐level precision and identification via score_user().
    """
    model.eval()

    # --- 1) Precompute per‐user enrollment & verification embeddings ---
    enr_embs = {}
    ver_embs = {}
    precisions = {}

    for u, seq_indices in val_user_to_indices.items(): # For each user
        user_seqs = [val_sequences[i] for i in seq_indices] # Their sequences (200, 46)
        all_emb = embed_sessions( # Getting the embeddings of all the user's sequences
            model=model,
            val_sequences=user_seqs,
            val_user_ids=[u] * len(user_seqs),
            device=device,
            batch_size=batch_size,
        )  # (n_samples_of_user, d_model)

        split_pt = len(all_emb) // 2 # Splitting into enrollment and verification by half
        e_emb = all_emb[:split_pt].clone()   # (n_enroll_u, d_model)
        v_emb = all_emb[split_pt:].clone()   # (n_verify_u, d_model)

        # Storing in dictionary
        enr_embs[u] = e_emb
        ver_embs[u] = v_emb

        centered = e_emb - e_emb.mean(dim=0, keepdim=True)  # (n_enroll_u, d_model)
        P = torch.tensor(
            LedoitWolf()
            .fit(centered.cpu().numpy())
            .precision_
            .astype(np.float32)
        )  # (d_model, d_model)
        precisions[u] = P

    all_users = list(val_user_to_indices.keys()) # List of all the users
    results = {}

    # Loop over each desired group size
    for k in group_sizes:
        mah_eer_list = [] # Authentication metric - Maha. EER
        id_acc_list = [] # Identification metric

        group_combinations_count = 0
        for group in itertools.combinations(all_users, k):
            group_combinations_count += 1

            group = list(group)  # User ID's of those user's in the group

             # --- 2.1) Build pooled genuine & impostor Mahalanobis scores for this group ---
            genuine_scores = []
            impostor_scores = []

            # Pre‐compute "outside‐group" verification embeddings once
            outside_users = [w for w in all_users if w not in group]
            out_ver_list = [enr_embs[w] for w in outside_users]
            out_ver_all = torch.cat(out_ver_list, dim=0)  # (sum_{w∉G} n_verify_w, d_model)

            # For each user u in G, gather:
            #   - genuine: Mahalanobis(e_emb_u, v_emb_u, P_u)
            #   - impostor: Mahalanobis(e_emb_u, out_ver_all, P_u)
            for u in group:
                e_u = enr_embs[u]          # (n_enroll_u, d_model)
                v_u = ver_embs[u]          # (n_verify_u, d_model)
                P_u = precisions[u]        # (d_model, d_model)

                # 2.1a) Genuine scores for u
                _, mah_g = score_user(e_u, v_u, P_u, get_cosine_score=False)
                genuine_scores.append(mah_g)

                # 2.1b) Impostor scores for u (all outside‐group ver vs. enroll_u)
                _, mah_i = score_user(e_u, out_ver_all, P_u, get_cosine_score=False)
                impostor_scores.append(mah_i)


            # Concatenate across all u∈G
            genuine_scores = np.concatenate(genuine_scores, axis=0)
            impostor_scores = np.concatenate(impostor_scores, axis=0)

            # 2.1c) Compute a single Mahalanobis‐based EER for group G
            y_true = np.concatenate(
                [np.ones_like(genuine_scores), np.zeros_like(impostor_scores)]
            )
            y_score = np.concatenate([genuine_scores, impostor_scores])
            eer_m, _thr_m, _ = compute_eer(y_true, y_score)
            mah_eer_list.append(eer_m)


            # Identification accuracy within group G
            # For each user v ∈ G, we will need (enroll_emb_v, P_v) = (enrollment embeddings of user, covariance matrix for maha. calc.).
            per_user_model = {}
            for v in group:
                e_v = enr_embs[v] # (n_enroll_v, d_model)
                P_v = precisions[v]
                per_user_model[v] = (e_v, P_v)

            correct = 0
            total = 0

            # Now, for each true user u ∈ G, and for each verification embedding x ∈ ver_embs[u]:
            #   - We will call score_user(enroll_v, x_batch, P_v) for each candidate v ∈ G
            #   - That returns an array of Mahalanobis scores (length = batch size = 1 if we pass one sample)
            #   - We compare those k Mahalanobis scores, pick the argmax, and see if it equals u's index in group.
            for idx_u, u in enumerate(group):
                # User's verification embeddings
                v_u = ver_embs[u]  # (n_ver_u, d_model)

                # Size of the verification embedding 
                n_ver_u = v_u.size(0)

                # Gather Mahalanobis scores for all k candidates over the entire v_u batch at once.
                # mah_matrix will be (n_ver_u, k) 
                mah_matrix = torch.zeros((n_ver_u, k))

                # For each candidate user
                for idx_v, v in enumerate(group):

                    # Enrollment embeddings and covariance of this user
                    e_v, P_v = per_user_model[v]

                    _, mah_vu = score_user(e_v, v_u, P_v, get_cosine_score=False)
                    # mah_vu is a NumPy 1D array of length n_ver_u; convert to tensor:
                    mah_matrix[:, idx_v] = torch.from_numpy(mah_vu)

                # Now mah_matrix[i,j] is Mahalanobis score of v_u[i] under candidate v_j.
                # We pick argmax over j for each i:
                preds = mah_matrix.argmax(dim=1).numpy()  # shape (n_ver_u,)

                # Count how many predictions match idx_u
                correct += int((preds == idx_u).sum())
                total += n_ver_u

            if total > 0:
                id_acc = correct / total * 100.0
                id_acc_list.append(id_acc)

        # 2.b) After iterating all groups of size k, average results
        avg_mah_eer = float(np.mean(mah_eer_list)) if mah_eer_list else None
        avg_id_acc  = float(np.mean(id_acc_list))  if id_acc_list  else None

        results[k] = {"mah_eer": avg_mah_eer, "id_acc": avg_id_acc, "count": group_combinations_count}

    model.train()
    return results


def plot_tsne(embeddings: np.ndarray,
              labels: np.ndarray,
              max_per_label = 100,
              title: str = "t-SNE (2D) of Embeddings") -> None:


    unique_labels = np.unique(labels)

    for group in itertools.combinations(list(unique_labels), 2):
        print("type(group)", type(group))
        group = list(group)

        unique_labels = np.unique(labels)
        selected_indices = []

        for u in group:
            user_idx = np.where(labels == u)[0]
            if len(user_idx) > max_per_label:
                chosen = user_idx[:max_per_label]
                # randomly pick max_per_user indices for this user
                # chosen = rng.choice(user_idx, size=50, replace=False)
            else:
                # keep all if ≤ max_per_user
                chosen = user_idx
            selected_indices.append(chosen)

         # concatenate all chosen indices and shuffle them
        selected_indices = np.concatenate(selected_indices)

        # subset embeddings and labels
        emb_relevant = embeddings[selected_indices]
        lab_relevant = labels[selected_indices]
        
        tsne = TSNE(n_components=2,init="pca", random_state=42)
        embs_2d = tsne.fit_transform(emb_relevant)


        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embs_2d[:, 0],
            embs_2d[:, 1],
            c=lab_relevant,
            cmap="tab20",
            s=50,
            alpha=0.8,
            linewidths=0.5,
        )
        plt.title(title)
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        plt.legend(*scatter.legend_elements(), title="User ID", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

def plot_tsne_with_ellipses(embeddings: np.ndarray,
                            labels: np.ndarray,
                            max_per_label: int = 100,
                            title: str = "t-SNE (2D) of Embeddings with Ellipses") -> None:
    """
    For every pair of distinct labels, take up to `max_per_label` samples of each,
    run t-SNE → 2D, and plot:
      (1) a scatter colored by label, and
      (2) 2σ covariance‐based ellipses around each class in the t-SNE plane.
    """
    unique_labels = np.unique(labels)

    for group in itertools.combinations(list(unique_labels), 2):
        group = list(group)
        # --- Select up to max_per_label indices per label in this pair ---
        selected_indices = []
        for u in group:
            user_idx = np.where(labels == u)[0]
            if len(user_idx) > max_per_label:
                chosen = user_idx[:max_per_label]
            else:
                chosen = user_idx
            selected_indices.append(chosen)

        # Concatenate & shuffle
        selected_indices = np.concatenate(selected_indices)

        emb_relevant = embeddings[selected_indices]  # shape: (≤2*max_per_label, d)
        lab_relevant = labels[selected_indices]      # shape: (same)

        # --- Run t-SNE on emb_relevant ---
        tsne = TSNE(n_components=2, init="pca", random_state=42)
        embs_2d = tsne.fit_transform(emb_relevant)  # shape: (n_sel, 2)

        # --- Compute mean & covariance in 2D for each class in this pair ---
        means_2d = {}
        covs_2d  = {}
        for lbl in group:
            pts_2d = embs_2d[lab_relevant == lbl]
            means_2d[lbl] = pts_2d.mean(axis=0)
            # 2D covariance of the t-SNE points for this class
            covs_2d[lbl]  = np.cov(pts_2d, rowvar=False)

        # --- Helper to draw a 2σ ellipse in 2D ---
        def draw_ellipse(ax, center, cov2, n_std=2.0, **kwargs):
            eigvals, eigvecs = np.linalg.eigh(cov2)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width   = 2 * n_std * np.sqrt(eigvals[0])
            height  = 2 * n_std * np.sqrt(eigvals[1])
            ellipse = Ellipse(xy=center,
                              width=width,
                              height=height,
                              angle=angle,
                              **kwargs)
            ax.add_patch(ellipse)

        # --- Plotting: TSNE scatter + ellipses ---
        plt.figure(figsize=(8, 6))

        # Scatter of all points in this pair
        scatter = plt.scatter(
            embs_2d[:, 0],
            embs_2d[:, 1],
            c=lab_relevant,
            cmap="tab20",
            s=50,
            alpha=0.75,
            linewidths=0.5,
        )
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")

        # Draw 2σ ellipse for each class
        colors = [plt.cm.tab10(i) for i in group]
        for lbl, color in zip(group, colors):
            draw_ellipse(plt.gca(),
                         means_2d[lbl],
                         covs_2d[lbl] + 1e-6 * np.eye(2),  # small reg to avoid singular
                         n_std=2.0,
                         edgecolor=color,
                         facecolor="none",
                         lw=1.5)

        plt.legend(
            *scatter.legend_elements(),
            title="Label",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.show()

        
def compute_silhouette(embeddings: np.ndarray,
                       labels: np.ndarray) -> float:
    """
    Return the silhouette score for `embeddings` (shape = [n_samples, d]) 
    with integer cluster labels `labels` (shape = [n_samples]). If fewer 
    than 2 unique labels appear, returns np.nan.
    """
    unique_labels = np.unique(labels)
    print("Unique Labels", unique_labels)
    if len(unique_labels) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels))
