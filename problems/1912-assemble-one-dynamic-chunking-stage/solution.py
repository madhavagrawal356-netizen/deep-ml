import torch
import torch.nn as nn
import torch.nn.functional as F


class DCStage(nn.Module):
    def __init__(self, d_model, main):
        """W_q and W_k start as the identity, residual starts at zero; all three are bias-free."""
        super().__init__()
        super().__init__()

        self.main = main

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        self.residual = nn.Linear(d_model, d_model, bias=False)

        # Initialize W_q and W_k as identity matrices
        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)

        # Initialize residual weights to zero
        nn.init.zeros_(self.residual.weight)

    def forward(self, x_hat):
        """x_hat (B, L, D). Returns z (B, L, D), p float (B, L), b bool (B, L)."""
        B, L, D = x_hat.shape

        # 1. Route
        q = self.W_q(x_hat)
        k = self.W_k(x_hat)

        similarity = F.cosine_similarity(
            q[:, 1:, :], k[:, :-1, :], dim=-1
        )

        p = torch.cat([
            torch.ones(B, 1, device=x_hat.device, dtype=x_hat.dtype),
            (1 - similarity) / 2
        ], dim=1).clamp(0, 1)

        b = p >= 0.5

        # 2. Compact
        counts = b.sum(dim=1)
        M = int(counts.max().item())

        positions = torch.cumsum(b.long(), dim=1) - 1
        compact = x_hat.new_zeros(B, M, D)
        compact_p = p.new_zeros(B, M)

        batch_idx, time_idx = torch.where(b)
        compact_idx = positions[batch_idx, time_idx]

        compact = compact.index_put(
            (batch_idx, compact_idx),
            x_hat[batch_idx, time_idx]
        )

        compact_p = compact_p.index_put(
            (batch_idx, compact_idx),
            p[batch_idx, time_idx]
        )

        mask = (
            torch.arange(M, device=x_hat.device)[None, :]
            < counts[:, None]
        )

        # 3. Main network
        z_hat = self.main(compact)

        # 4. Smooth
        smoothed = []
        prev = torch.zeros(B, D, device=x_hat.device, dtype=x_hat.dtype)

        for t in range(M):
            pt = compact_p[:, t].unsqueeze(-1)
            prev = pt * z_hat[:, t, :] + (1 - pt) * prev
            smoothed.append(prev)

        smoothed = torch.stack(smoothed, dim=1)

        # 5. Upsample
        z = torch.gather(
            smoothed,
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, D)
        )

        # 6. Confidence (forward value = 1, gradient = gradient of c)
        c = torch.where(b, p, 1 - p)
        factor = 1.0 + c - c.detach()
        z = z * factor.unsqueeze(-1)

        # 7. Residual
        z = z + self.residual(x_hat)

        return z, p, b
