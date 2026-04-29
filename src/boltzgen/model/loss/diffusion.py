# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

import einx
import torch
import torch.nn.functional as F
from einops import einsum, rearrange


def weighted_rigid_centering(
    true_coords,  # Float['b n 3'],       #  true coordinates
    pred_coords,  # Float['b n 3'],       # predicted coordinates
    weights,  # Float['b n'],             # weights for each atom
    mask,  # Bool['b n'] | None = None    # mask for variable lengths
):  # -> Float['b n 3']:
    """Algorithm 28  without rotation alignment"""
    # zero out all predicted and true coordinates where not an atom
    mask = mask.bool()
    true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
    pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
    weights = einx.where("b n, b n, -> b n", mask, weights, 0.0)

    # Take care of weights broadcasting for coordinate dimension
    weights = rearrange(weights, "b n -> b n 1")

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid

    # Apply the translation
    aligned_coords = true_coords_centered + pred_centroid
    aligned_coords.detach_()

    return aligned_coords


def _det3x3(m: torch.Tensor) -> torch.Tensor:
    """Closed-form 3×3 determinant using only element-wise ops (HPU-native).

    Avoids torch.det / aten::_linalg_det which falls back to CPU on HPU.
    Inputs: m shape (..., 3, 3).  Returns shape (...,).
    """
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _svd3x3_jacobi(A: torch.Tensor, n_iter: int = 8) -> tuple:
    """One-sided Jacobi SVD for batched 3×3 matrices using HPU-native ops.

    Replaces torch.linalg.svd (aten::_linalg_svd) which falls back to CPU on HPU.
    Computes A = U @ diag(S) @ V^T for each matrix in the batch.

    Algorithm: iterative Jacobi sweeps on A^T A to find V (right singular vectors),
    then U = A @ V / S.  Converges in ~8 iterations for well-conditioned 3×3 matrices.

    Parameters
    ----------
    A : Tensor of shape (batch, 3, 3)
    n_iter : int, number of Jacobi sweep iterations (default 8, sufficient for 3×3)

    Returns
    -------
    U : (batch, 3, 3) — left singular vectors
    S : (batch, 3)    — singular values (non-negative, not necessarily sorted)
    V : (batch, 3, 3) — right singular vectors  (NOT transposed, i.e. A = U S V^T)
    """
    batch = A.shape[0]
    device, dtype = A.device, A.dtype

    # B = A^T A  (symmetric positive semi-definite)
    B = torch.bmm(A.transpose(-1, -2), A)  # (batch, 3, 3)

    # Accumulate V via Jacobi rotations applied to B
    V = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1).clone()

    # Jacobi sweep over all off-diagonal pairs (p,q): (0,1),(0,2),(1,2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for _ in range(n_iter):
        for p, q in pairs:
            # Compute Jacobi rotation angle for the (p,q) element
            Bpp = B[:, p, p]
            Bqq = B[:, q, q]
            Bpq = B[:, p, q]

            # For G = [[c,s],[-s,c]] convention (G[p,q]=s, G[q,p]=-s), the Jacobi
            # rotation that zeros out B[p,q] requires:
            #   B'[p,q] = c*s*(Bpp - Bqq) + (c²-s²)*Bpq = 0
            #   → tan(2θ) = -2*Bpq / (Bpp - Bqq)
            denom = Bpp - Bqq
            theta = 0.5 * torch.atan2(-2.0 * Bpq, denom + 1e-30)
            c = torch.cos(theta)  # (batch,)
            s = torch.sin(theta)  # (batch,)

            # Build Jacobi rotation matrix G (identity with (p,p)=c, (q,q)=c,
            # (p,q)=s, (q,p)=-s)
            G = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1).clone()
            G = G.clone()
            G[:, p, p] = c
            G[:, q, q] = c
            G[:, p, q] = s
            G[:, q, p] = -s

            # Update B = G^T B G  and  V = V G
            B = torch.bmm(torch.bmm(G.transpose(-1, -2), B), G)
            V = torch.bmm(V, G)

    # Singular values = sqrt of diagonal of B (= eigenvalues of A^T A)
    S = torch.clamp(torch.diagonal(B, dim1=-2, dim2=-1), min=0.0).sqrt()  # (batch, 3)

    # U = A V / ||AV_col||  — normalize by actual column norms for orthogonality.
    # Using Jacobi-computed S causes instability when singular values differ greatly.
    AV = torch.bmm(A, V)  # (batch, 3, 3); columns should be orthogonal if V is
    col_norms = AV.norm(dim=-2, keepdim=True).clamp(min=1e-10)  # (batch, 1, 3)
    U = AV / col_norms  # (batch, 3, 3) with unit-norm orthogonal columns

    return U, S, V


def weighted_rigid_align(
    true_coords,  # Float['b n 3'],       # true coordinates
    pred_coords,  # Float['b n 3'],       # predicted coordinates
    weights,  # Float['b n'],             # weights for each atom
    mask,  # Bool['b n'] | None = None    # mask for variable lengths
):  # -> Float['b n 3']:
    """Algorithm 28 : note there is a problem with the pseudocode in the paper where predicted and
    GT are swapped in algorithm 28, but correct in equation (2). Aligns true_coords to pred_coords.
    """
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    # HPU lazy mode: avoid bool(tensor) as Python if condition — forces graph flush
    if torch.any(mask.sum(dim=-1) < (dim + 1)).cpu().item():
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
             "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)

    # HPU does not support torch.linalg.svd or torch.det natively — both fall back
    # to CPU causing 8000+ HPU→CPU→HPU round trips per run (4000 SVD + 4000 det).
    # Replace with HPU-native implementations:
    #   SVD: one-sided Jacobi iterations on the 3×3 covariance matrix
    #   det: closed-form 3×3 determinant using element-wise ops only
    if cov_matrix_32.device.type == "hpu":
        U, S, V = _svd3x3_jacobi(cov_matrix_32)
    else:
        U, S, V = torch.linalg.svd(
            cov_matrix_32,
            driver="gesvd" if cov_matrix_32.device.type == "cuda" else None,
        )
        V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    # HPU lazy mode: avoid bool(tensor) — move check to CPU side using .item()
    if (S.abs() <= 1e-15).any().cpu().item() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
             "cross-correlation between aligned point clouds. "
             "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1.
    # HPU: use closed-form 3×3 det (element-wise) instead of torch.det (CPU fallback).
    if rot_matrix.device.type == "hpu":
        det_vals = _det3x3(rot_matrix)
    else:
        det_vals = torch.det(rot_matrix)
    ones = torch.ones(batch_size, dim - 1, dtype=cov_matrix_32.dtype, device=cov_matrix.device)
    diag_vals = torch.cat([ones, det_vals.unsqueeze(-1)], dim=-1)  # (batch, dim)
    F = torch.diag_embed(diag_vals)  # (batch, dim, dim)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords = aligned_coords.detach()

    return aligned_coords


def smooth_lddt_loss(
    pred_coords,  # Float['b n 3'],
    true_coords,  # Float['b n 3'],
    is_nucleotide,  # Bool['b n'],
    coords_mask,  # Bool['b n'] | None = None,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):  # -> Float['']:
    """Algorithm 27
    pred_coords: predicted coordinates
    true_coords: true coordinates
    Note: for efficiency pred_coords is the only one with the multiplicity expanded
    TODO: add weighing which overweight the smooth lddt contribution close to t=0 (not present in the paper)
    """
    lddt = []
    for i in range(true_coords.shape[0]):
        true_dists = torch.cdist(true_coords[i], true_coords[i])

        is_nucleotide_i = is_nucleotide[i // multiplicity]
        coords_mask_i = coords_mask[i // multiplicity]

        is_nucleotide_pair = is_nucleotide_i.unsqueeze(-1).expand(
            -1, is_nucleotide_i.shape[-1]
        )

        mask = is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        mask += (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
        mask *= 1 - torch.eye(pred_coords.shape[1], device=pred_coords.device)
        mask *= coords_mask_i.unsqueeze(-1)
        mask *= coords_mask_i.unsqueeze(-2)

        valid_pairs = mask.nonzero()
        true_dists_i = true_dists[valid_pairs[:, 0], valid_pairs[:, 1]]

        pred_coords_i1 = pred_coords[i, valid_pairs[:, 0]]
        pred_coords_i2 = pred_coords[i, valid_pairs[:, 1]]
        pred_dists_i = F.pairwise_distance(pred_coords_i1, pred_coords_i2)

        dist_diff_i = torch.abs(true_dists_i - pred_dists_i)

        eps_i = (
            F.sigmoid(0.5 - dist_diff_i)
            + F.sigmoid(1.0 - dist_diff_i)
            + F.sigmoid(2.0 - dist_diff_i)
            + F.sigmoid(4.0 - dist_diff_i)
        ) / 4.0

        lddt_i = eps_i.sum() / (valid_pairs.shape[0] + 1e-5)
        lddt.append(lddt_i)

    # average over batch & multiplicity
    return 1.0 - torch.stack(lddt, dim=0).mean(dim=0)

def compute_bond_loss(pred_atom_coords, true_coords, feats):
    bond_loss = torch.zeros(pred_atom_coords.shape[0], device=pred_atom_coords.device)
    num_bonds = torch.tensor(0, device=pred_atom_coords.device)
    for index_batch in range(len(feats["connections_edge_index"])):
        if feats["connections_edge_index"][index_batch].shape[1] == 0:
            continue
        pred_bond_coords = pred_atom_coords[
            :, feats["connections_edge_index"][index_batch]
        ]
        true_bond_coords = pred_atom_coords[
            :, feats["connections_edge_index"][index_batch]
        ]
        pred_bond_lengths = torch.linalg.norm(
            pred_bond_coords[:, 0] - pred_bond_coords[:, 1], dim=-1
        )
        true_bond_lengths = torch.linalg.norm(
            true_bond_coords[:, 0] - true_bond_coords[:, 1], dim=-1
        )
        bond_loss += torch.sum((pred_bond_lengths - true_bond_lengths) ** 2, dim=1)
        num_bonds += pred_bond_lengths.shape[1]
    if num_bonds > 0:
        bond_loss /= num_bonds
    return bond_loss, num_bonds
