"""Oriented-cuboid primitive and fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

if TYPE_CHECKING:
    from ._config import SpherizeParams

PrimitiveMode = Literal["sphere", "cuboid", "auto"]
"""Which primitive a link is approximated with."""


@jdc.pytree_dataclass
class Cuboid:
    """An oriented box in the link frame.

    The box is ``{center + R(wxyz) @ (h * u) : u in [-1, 1]^3}``, where ``h`` is
    ``half_extents``. Single cuboid: center (3,), half_extents (3,), wxyz (4,).
    """

    center: jnp.ndarray
    half_extents: jnp.ndarray
    wxyz: jnp.ndarray

    @property
    def volume(self) -> float:
        return float(8.0 * np.prod(np.asarray(self.half_extents)))

    def rotation_matrix(self) -> np.ndarray:
        return _wxyz_to_matrix(np.asarray(self.wxyz))


def _wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = w * w + x * x + y * y + z * z
    w, x, y, z = (v / np.sqrt(n) for v in (w, x, y, z))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def make_cuboid(center: np.ndarray, half_extents: np.ndarray, R: np.ndarray) -> Cuboid:
    """Build a Cuboid from a rotation matrix (reflections are fixed up)."""
    R = np.array(R, dtype=float)
    if np.linalg.det(R) < 0:
        # A box is symmetric, so flipping one axis gives the same box.
        R[:, 0] *= -1.0
    return Cuboid(
        center=jnp.asarray(center),
        half_extents=jnp.asarray(half_extents),
        wxyz=jnp.asarray(_matrix_to_wxyz(R)),
    )


def cuboid_sdf(points: jnp.ndarray, center, half_extents, wxyz) -> jnp.ndarray:
    """Signed distance from points (N, 3) to an oriented box (negative inside).

    Differentiable w.r.t. all arguments (jax).
    """
    # Rotation matrix from quaternion (jax-friendly).
    q = wxyz / jnp.linalg.norm(wxyz)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = (jnp.atleast_2d(points) - center) @ R  # == R^T (p - c) per row
    q_ = jnp.abs(local) - half_extents
    outside = jnp.sqrt(jnp.sum(jnp.maximum(q_, 0.0) ** 2, axis=-1) + 1e-12)
    inside = jnp.minimum(jnp.max(q_, axis=-1), 0.0)
    return outside + inside


def transform_cuboid(cuboid: Cuboid, T: np.ndarray) -> Cuboid:
    """Apply a 4x4 transform (rotation, mirror and/or translation) to a cuboid."""
    T = np.asarray(T)
    center = T[:3, :3] @ np.asarray(cuboid.center) + T[:3, 3]
    R = T[:3, :3] @ cuboid.rotation_matrix()
    return make_cuboid(center, np.asarray(cuboid.half_extents), R)


def fit_cuboid(points: np.ndarray, padding: float = 1.0) -> Cuboid:
    """Fit a (near) minimum-volume oriented box around points.

    Tries the axis-aligned frame and the PCA frame and keeps the smaller box.
    Axis-aligned is tried because PCA axes are arbitrary for symmetric shapes
    (e.g. a cube), which would otherwise give a needlessly rotated box.
    """
    if len(points) == 0:
        return make_cuboid(np.zeros(3), np.full(3, 1e-2), np.eye(3))

    mean = points.mean(axis=0)
    candidates = [np.eye(3)]
    if len(points) >= 4:
        _, _, Vt = np.linalg.svd(points - mean, full_matrices=False)
        if Vt.shape[0] == 3:
            candidates.append(Vt.T)

    best = None
    for R in candidates:
        local = (points - mean) @ R
        lo, hi = local.min(axis=0), local.max(axis=0)
        half = np.maximum((hi - lo) / 2.0 * padding, 1e-4)
        vol = float(np.prod(half))
        # Prefer axis-aligned unless PCA is meaningfully tighter.
        if best is None or vol < best[0] * 0.98:
            best = (vol, R, mean + R @ ((lo + hi) / 2.0), half)
    assert best is not None
    _, R, center, half = best
    return make_cuboid(center, half, R)


def _tightness(points: np.ndarray, cuboid: Cuboid) -> float:
    """Box volume / convex hull volume of points."""
    if len(points) < 4:
        return 1.0
    try:
        hull_vol = ConvexHull(points).volume
    except (QhullError, ValueError):
        return 1.0
    return float(cuboid.volume / (hull_vol + 1e-10))


def cuboidize(
    mesh: trimesh.Trimesh,
    target_cuboids: int,
    params: "SpherizeParams | None" = None,
) -> list[Cuboid]:
    """Approximate a mesh with up to ``target_cuboids`` oriented boxes.

    Fits an oriented box to the surface samples and, while the budget allows and
    the box is loose relative to the convex hull (``target_tightness``), splits
    along the box's longest axis and recurses. Like ``spherize`` the result may
    have fewer boxes than the budget when a single box is already tight.

    Uses ``n_samples``, ``padding`` and ``target_tightness`` from SpherizeParams.
    """
    from ._config import SpherizeParams

    p = params or SpherizeParams()
    points = cast(np.ndarray, mesh.sample(p.n_samples))

    def split(pts: np.ndarray, budget: int) -> list[Cuboid]:
        if len(pts) == 0 or budget <= 0:
            return []
        box = fit_cuboid(pts, p.padding)
        if budget <= 1 or len(pts) < 20 or _tightness(pts, box) <= p.target_tightness:
            return [box]

        # Split along the longest box axis, blending median and midpoint.
        R = box.rotation_matrix()
        axis = R[:, int(np.argmax(np.asarray(box.half_extents)))]
        proj = pts @ axis
        split_at = 0.5 * np.median(proj) + 0.5 * (proj.min() + proj.max()) / 2
        left, right = pts[proj <= split_at], pts[proj > split_at]
        if len(left) < 10 or len(right) < 10:
            return [box]

        left_budget = min(budget - 1, max(1, round(budget * len(left) / len(pts))))
        out = split(left, left_budget) + split(right, budget - left_budget)
        return out or [box]

    return split(points, target_cuboids)
