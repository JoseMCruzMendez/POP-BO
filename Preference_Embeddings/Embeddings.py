import torch
import torch.nn as nn
from torch.nn.utils import parametrizations
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

def left_mult_by_J(v: torch.Tensor) -> torch.Tensor:
    """
    Helper function to efficiently implement multiplication by the block matrix J in O(n) instead of O(n^2)
    """
    *batch_dims, n = v.shape
    k = n // 2

    # reshape to [..., k, 2]
    V = v.view(*batch_dims, k, 2)

    # rotate each 2-vector [x,y] -> [-y, x]
    V_rot = torch.stack(
        (-V[..., 1],   # –y
         V[..., 0]),  #  x
        dim=-1        # put the 2 back into the last axis
    )
    # flatten back to [..., 2k]
    return V_rot.view_as(v)


class Embedding(nn.Module):
    def __init__(self, in_dim, n, sizes = None):
        super().__init__()
        if sizes is None:
            sizes = [128, 64]
        layers = []
        last_dim = in_dim
        for h in sizes:
            layers += [nn.Linear(last_dim, h), nn.Hardswish()]
            last_dim = h
        layers += [nn.Linear(last_dim, n)]
        self.embedding = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.embedding(x)

class PreferenceEmbedding(nn.Module):
    def __init__(self, in_dim,  factor = 2, sizes = None):
        """
        Args:
          n (int): dimension of input vectors x, x'
          J (Tensor[n,n]): constant matrix in the bilinear form
        """
        super().__init__()
        assert factor % 2 == 0, "factor must be even"
        n = in_dim * factor #embeddings dimension must be even
        J = torch.eye(n//2).kron(torch.tensor([[0, -1], [1, 0]]))
        self.register_buffer('J', J)
        # 1) Declare an unconstrained weight
        self.embedding = Embedding(in_dim, n, sizes = sizes)
        self.U = nn.Parameter(torch.randn(n, n))
        # 2) Apply an orthogonal re-parametrization
        parametrizations.orthogonal(self, 'U')
        # 3) Store J as a buffer

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
        JUxp = left_mult_by_J(Uxp)                    # [B, n]
        w   = (Ux * JUxp).sum(dim=1)          # [B]
        return w               # [B]


class ComplexPreference(nn.Module):
    """Model that implements the preference score as the imaginary part of the Hermitian Inner Product of two embedded vectors.
    Comes from Section A.1 of the paper General Preference Modeling with Preference Representations for Aligning Language Models"""
    def __init__(self, in_dim, factor = 2, sizes = None, branches = None):
        super().__init__()
        assert factor % 2 == 0, "factor must be even"
        self.hparams = {"in_dim": in_dim, "factor": factor, "sizes": sizes, "branches": branches}
        n = in_dim * factor #embeddings dimension must be even
        #self.init_embedding = Embedding(in_dim, n)
        #last size is differentiating layer
        if branches is None:
            branch_size = 128
            branches = 1
            radii_features = nn.Linear(branch_size, n//2)
            angles_features = nn.Linear(branch_size, n//2)
        else:
            branch_size = sizes[-branches]
            radii_features = Embedding(branch_size, n//2, sizes=sizes[-branches:])
            angles_features = Embedding(branch_size, n//2, sizes=sizes[-branches:])
        if sizes is not None:
            sizes = sizes[:-branches]
            self.trunk = nn.Sequential(
                Embedding(in_dim, branch_size, sizes=sizes),
                nn.Hardswish()
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, branch_size),
                nn.Hardswish()
            )
        #divide everything by 2 since there are two networks
        self.radii = nn.Sequential(
            self.trunk,
            nn.Hardswish(),
            radii_features,
            nn.Softplus()
        )
        self.angles = nn.Sequential(
            self.trunk,
            nn.Hardswish(),
            angles_features,
            nn.Softplus()
        )

    def forward(self, x, x_prime):
        x_radii = self.radii(x)
        x_angles = self.angles(x) * 2 * torch.pi
        x_prime_radii = self.radii(x_prime)
        x_prime_angles = self.angles(x_prime) * 2 * torch.pi
        im_part = torch.sum(x_radii * x_prime_radii * torch.sin(x_angles - x_prime_angles), dim=1)
        return im_part


class EigenPreference(nn.Module):
    """Model that implements the preference score using an eigenvalue layer and an eigenvector layer, as described in
    Section A.2 of the paper General Preference Modeling with Preference Representations for Aligning Language Models"""
    def __init__(self, in_dim, factor=2, sizes=None):
        super().__init__()
        assert factor % 2 == 0, "factor must be even"
        n = in_dim * factor #embeddings dimension must be even
        #In the paper, eigenvalues are prompt-dependent. We have a "same prompt" approach as we are maximizing
        #Over the same function in the toy example, so eigenvalues should be input-independent as inputs are responses
        self.eigenvalue_layer = nn.Parameter(torch.randn(n//2))
        self.eigenvector_layer = Embedding(in_dim, n, sizes=sizes)

    def forward(self, x, x_prime):
        #First, calculate embeddings
        x, x_prime = self.eigenvector_layer(x), self.eigenvector_layer(x_prime) #[B,n]
        #normalize according to paper
        x = x / torch.norm(x, dim=1, keepdim=True)
        x_prime = x_prime / torch.norm(x_prime, dim=1, keepdim=True)
        #Second, calculate eigenvalue Matrix D
        eigenvals = torch.exp(self.eigenvalue_layer) #[n//2]
        D_diag = eigenvals.repeat_interleave(2).sqrt() #[n]
        #Since inner product is x^T DRD xp we calculate as (D^Tx) R (Dxp) for O(n) computations
        #Now transpose before doing multiplications
        Dx = D_diag * x #[B, n]
        Dxp = D_diag * x_prime #[B, n]
        JDxp = left_mult_by_J(Dxp) #[B, n]
        #return the raw score for sigmoid trick in loss function for numerical stability
        score = (Dx * JDxp).sum(dim=1) #[B]
        return score





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
    def __init__(self, num_pairs, func, minimize = True, in_dim = 2, seed=None, bounds=None):
        self.num_pairs = num_pairs
        # sample points once; you could re-sample each epoch if you like
        if seed is not None:
            torch.manual_seed(seed)
        if bounds is None:
            bounds = ((0,1), (0,1))
        self.X = torch.empty(num_pairs, in_dim)
        self.Y = torch.empty(num_pairs, in_dim)
        for dim, dim_data in enumerate(bounds):
            dist = torch.distributions.Uniform(dim_data[0], dim_data[1])
            self.X[:,dim] = dist.sample((num_pairs,))
            self.Y[:,dim] = dist.sample((num_pairs,))
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

def evaluate(model, func, bounds=None, device='cpu'):
    dataset = FuncDataset(num_pairs=50_000, func = func, minimize=True, in_dim=2, bounds=bounds)
    test_loader  = DataLoader(dataset, batch_size=512, shuffle=True)
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

def train_on_func(model, func, bounds=None, epochs=10, lr=1e-1, patience=10):
    # 5) DataLoader
    dataset = FuncDataset(num_pairs=50_000, func = func, minimize=True, in_dim=2, bounds=bounds)
    loader  = DataLoader(dataset, batch_size=512, shuffle=True)
    total_steps = epochs * len(loader)
    warmup_steps = int(0.1 * total_steps)

    #optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            # cosine decay after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = torch.inf
    bad_epochs = 0
    for epoch in range(1, epochs):

        for X, Xp, Y in loader:
            optimizer.zero_grad()
            preds = model(X, Xp)
            loss  = loss_fn(preds, Y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_val = loss.detach().item()
        if loss_val < best_loss:
            best_loss = loss_val
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} — loss: {loss_val:.6f}")