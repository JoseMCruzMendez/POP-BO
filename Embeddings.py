import torch
import torch.nn as nn
from torch.nn.utils import parametrizations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

class Embedding(nn.Module):
    def __init__(self, in_dim, n):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            nn.Linear(in_dim, 4*n),
            nn.Hardswish(),
            nn.Linear(4*n, 2*n),
            nn.Hardswish(),
            nn.Linear(2*n, n)
        )
    def forward(self, x):
        return self.embedding(x)
class PreferenceEmbedding(nn.Module):
    def __init__(self, in_dim):
        """
        Args:
          n (int): dimension of input vectors x, x'
          J (Tensor[n,n]): constant matrix in the bilinear form
        """
        super().__init__()
        n = in_dim * 2 #embeddings dimension must be even
        J = torch.eye(n//2).kron(torch.tensor([[0, -1], [1, 0]]))
        # 1) Declare an unconstrained weight
        self.embedding = Embedding(in_dim, n)
        self.U = nn.Parameter(torch.randn(n, n))
        # 2) Apply an orthogonal re-parametrization
        parametrizations.orthogonal(self, 'U')
        # 3) Store J as a buffer
        self.register_buffer('J', J)

    def forward(self, x, x_prime):
        """Confirms whether x_prime is preferred to x."""
        # x, x_prime: [B, in_dim]
        x       = self.embedding(x)       # [B, 2*in_dim]
        x_prime = self.embedding(x_prime)   # [B, 2*in_dim]

        # U: [n, n] where n = 2*in_dim
        # apply U to each row of x and x_prime:
        Ux   = x       @ self.U.T          # [B, n]
        Uxp  = x_prime @ self.U.T          # [B, n]

        # compute batch of scalars w[i] = Uxp[i]^T J Ux[i]
        # J is [n, n]
        JUx = Ux @ self.J                     # [B, n]
        w   = (JUx * Uxp).sum(dim=1)          # [B]
        return torch.sigmoid(w)               # [B]


# 2) Quadrant‐comparison dataset
class QuadPairDataset(Dataset):
    def __init__(self, num_pairs):
        self.num_pairs = num_pairs
        # sample points once; you could re-sample each epoch if you like
        self.X = torch.randn(num_pairs, 2)
        self.Y = torch.randn(num_pairs, 2)
        # precompute labels
        self.labels = torch.zeros(num_pairs)
        for i in range(num_pairs):
            qx  = self.quadrant(self.X[i])
            qy  = self.quadrant(self.Y[i])
            # define cycle Q1>Q2>Q3>Q4>Q1
            # map: Q1=0, Q2=1, Q3=2, Q4=3
            #delta = (qy - qx) % 4
            # x > y if delta in {1,2}? Actually we want Qx > Qy if
            # moving from x to y you go forward less than 2 steps
            self.labels[i] = 1.0 if qx==3 and qy ==0 else qx < qy

    @staticmethod
    def quadrant(pt):
        x, y = pt
        if   x>=0 and y>=0: return 0   # Q1
        elif x < 0 <= y: return 1   # Q2
        elif x<0  and y<0 : return 2   # Q3
        else:                return 3   # Q4

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.labels[idx]

class ReshuffleQuadDataset(Dataset):
    def __init__(self, num_pairs, pool_size=10000, seed=None):
        """
        num_pairs:   nominal __len__ of the dataset (how many pairs you draw per epoch)
        pool_size:   how many base points to keep around
        seed:        optional RNG seed for reproducibility
        """
        super().__init__()
        self.num_pairs = num_pairs
        self.pool_size = pool_size
        if seed is not None:
            torch.manual_seed(seed)

        # sample a fixed pool of points once
        # shape: [pool_size, 2]
        self.points = torch.randn(pool_size, 2)

    @staticmethod
    def quadrant(pt):
        x, y = pt
        if   x >= 0 and y >= 0: return 0   # Q1
        elif x <  0 and y >= 0: return 1   # Q2
        elif x <  0 and y <  0: return 2   # Q3
        else:                   return 3   # Q4

    @staticmethod
    def label_from_quadrants(qx, qy):
        # your cycle Q1>Q2>Q3>Q4>Q1
        # return 1.0 if x > y else 0.0
        if qx == 3 and qy == 0:
            return 1.0
        return 1.0 if qx < qy else 0.0

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # ignore idx — we just draw a random pair each call
        i = torch.randint(0, self.pool_size, (1,)).item()
        j = torch.randint(0, self.pool_size, (1,)).item()
        # if you prefer no repeats, you can:
        #  while j == i:
        #      j = torch.randint(0, self.pool_size, (1,)).item()

        x       = self.points[i]
        x_prime = self.points[j]

        qx = self.quadrant(x)
        qy = self.quadrant(x_prime)
        label = self.label_from_quadrants(qx, qy)

        return x, x_prime, label

class FuncDataset(Dataset):
    def __init__(self, num_pairs, func, minimize = True, in_dim = 2, seed=None):
        self.num_pairs = num_pairs
        # sample points once; you could re-sample each epoch if you like
        if seed is not None:
            torch.manual_seed(seed)
        self.X = torch.randn(num_pairs, in_dim)
        self.Y = torch.randn(num_pairs, in_dim)
        # precompute labels
        self.labels = torch.zeros(num_pairs)
        self.func = func
        for i in range(num_pairs):
            qx  = self.func(self.X[i])
            qy  = self.func(self.Y[i])
            self.labels[i] = qx < qy if minimize else qx > qy
    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.labels[idx]

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for x, x_prime, y_true in test_loader:
            x, x_prime, y_true = x.to(device), x_prime.to(device), y_true.to(device)
            y_prob = model(x, x_prime)       # [B] float probabilities
            y_pred = (y_prob > 0.5).long()   # [B] binary predictions

            all_scores.append(y_prob.cpu())
            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu().long())

    # flatten lists
    scores  = torch.cat(all_scores).numpy()
    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # basic metrics
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average='binary'
    )
    cm = confusion_matrix(targets, preds)

    # optional: ROC-AUC
    try:
        auc = roc_auc_score(targets, scores)
    except ValueError:
        auc = float('nan')  # e.g. if one class missing

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
    }