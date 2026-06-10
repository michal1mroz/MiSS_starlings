from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.spatial import ConvexHull, KDTree, Delaunay
except ImportError:  # pragma: no cover
    ConvexHull = None
    KDTree = None
    Delaunay = None


def _normalize_positions(positions: Any) -> np.ndarray:
    """Convert position records to an array of shape (T, N, 3)."""
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[np.newaxis, ...]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError("positions must be a (N,3) or (T,N,3) array-like")


def compute_number_of_birds(positions: Any) -> int:
    """Return the number of birds in a position record."""
    arr = _normalize_positions(positions)
    return int(arr.shape[1])


def compute_velocities(positions: Any, dt: float = 1.0) -> np.ndarray:
    """Compute bird velocities from a sequence of position snapshots.

    Returns an array of shape (T-1, N, 3).
    """
    arr = _normalize_positions(positions)
    if arr.shape[0] < 2:
        return np.zeros((0, arr.shape[1], 3), dtype=np.float64)
    return np.diff(arr, axis=0) / float(dt)


def compute_mean_speed(positions: Any, dt: float = 1.0) -> float:
    """Compute the mean speed across birds and time from position records."""
    velocities = compute_velocities(positions, dt=dt)
    if velocities.size == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=-1)
    return float(np.mean(speeds))


def compute_centroid(snapshot: np.ndarray) -> np.ndarray:
    """Compute centroid of a single 3D snapshot."""
    return np.mean(snapshot, axis=0)


def compute_covariance(snapshot: np.ndarray) -> np.ndarray:
    """Compute covariance matrix of a 3D snapshot around its centroid."""
    centered = snapshot - compute_centroid(snapshot)
    return np.cov(centered, rowvar=False, bias=True)


def compute_principal_axes(snapshot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sorted principal axis lengths and directions.

    Returns:
      - axis_lengths: shape (3,) sorted ascending
      - axis_vectors: shape (3, 3) corresponding eigenvectors
    """
    cov = compute_covariance(snapshot)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigvals)
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]
    axis_lengths = np.sqrt(np.maximum(eigvals, 0.0))
    return axis_lengths, eigvecs


def compute_convex_hull_volume(snapshot: np.ndarray) -> float:
    """Compute convex hull volume of a point cloud.

    Falls back to axis-aligned bounding box volume when ConvexHull is unavailable.
    """
    if snapshot.shape[0] < 4:
        return 0.0
    if ConvexHull is not None:
        hull = ConvexHull(snapshot)
        return float(hull.volume)
    mins = np.min(snapshot, axis=0)
    maxs = np.max(snapshot, axis=0)
    return float(np.prod(maxs - mins))


def compute_alpha_shape_volume(snapshot: np.ndarray, alpha: float | None = None) -> float:
    """Approximate the 3D alpha-shape volume by summing volumes of Delaunay
    tetrahedra whose circumradius is below `alpha`.

    If `alpha` is None, a heuristic based on mean NND is used. Falls back to
    convex hull volume when Delaunay is unavailable or when the computation
    fails.
    """
    if snapshot.shape[0] < 4:
        return 0.0
    if Delaunay is None:
        return compute_convex_hull_volume(snapshot)

    try:
        if alpha is None:
            mnd = compute_nearest_neighbor_distance(snapshot)
            alpha = max(1e-6, 1.5 * float(mnd))

        delaunay = Delaunay(snapshot)

        def tetra_circumradius(p0, p1, p2, p3):
            A = np.vstack([p1 - p0, p2 - p0, p3 - p0]).T
            b = np.array([np.dot(p1, p1) - np.dot(p0, p0), np.dot(p2, p2) - np.dot(p0, p0), np.dot(p3, p3) - np.dot(p0, p0)])
            try:
                center = np.linalg.solve(2.0 * A, b)
            except np.linalg.LinAlgError:
                return float("inf")
            return float(np.linalg.norm(center - p0))

        total_vol = 0.0
        for tet in delaunay.simplices:
            p0, p1, p2, p3 = snapshot[tet]
            r = tetra_circumradius(p0, p1, p2, p3)
            if r <= alpha:
                vol = abs(np.linalg.det(np.stack([p1 - p0, p2 - p0, p3 - p0], axis=1))) / 6.0
                total_vol += vol
        if total_vol <= 0.0:
            return compute_convex_hull_volume(snapshot)
        return float(total_vol)
    except Exception:
        return compute_convex_hull_volume(snapshot)


def compute_flock_density(snapshot: np.ndarray, volume: float | None = None, use_alpha: bool = True) -> float:
    """Compute density as number of birds divided by flock volume.

    By default the flock volume is estimated with the alpha-shape (a-shape).
    """
    if volume is None:
        if use_alpha:
            volume = compute_alpha_shape_volume(snapshot)
        else:
            volume = compute_convex_hull_volume(snapshot)
    if volume <= 0.0:
        return 0.0
    return float(snapshot.shape[0] / volume)


def compute_nearest_neighbor_distance(snapshot: np.ndarray) -> float:
    """Compute the mean nearest-neighbor distance in a 3D snapshot."""
    if snapshot.shape[0] < 2:
        return 0.0
    if KDTree is not None:
        tree = KDTree(snapshot)
        distances, _ = tree.query(snapshot, k=2)
        nearest = distances[:, 1]
        return float(np.mean(nearest))
    diff = snapshot[:, np.newaxis, :] - snapshot[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(np.min(dists, axis=1)))


def compute_com_positions(positions: Any) -> np.ndarray:
    """Compute center-of-mass (mean position) for each snapshot.

    Returns an array of shape (T, 3).
    """
    arr = _normalize_positions(positions)
    return np.mean(arr, axis=1)


def compute_com_velocities(positions: Any, dt: float = 1.0) -> np.ndarray:
    """Compute velocities of the centre of mass across snapshots."""
    com = compute_com_positions(positions)
    if com.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    return np.diff(com, axis=0) / float(dt)


def compute_com_mean_speed(positions: Any, dt: float = 1.0) -> float:
    """Mean speed of the flock centre-of-mass (scalar)."""
    v = compute_com_velocities(positions, dt=dt)
    if v.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(v, axis=1)))


def compute_thickness_I1(snapshot: np.ndarray) -> float:
    """Compute the smallest principal axis thickness I1."""
    axis_lengths, _ = compute_principal_axes(snapshot)
    return float(axis_lengths[0])


def compute_aspect_ratios(snapshot: np.ndarray) -> tuple[float, float]:
    """Compute aspect ratios I2/I1 and I3/I1 from principal axis lengths."""
    axis_lengths, _ = compute_principal_axes(snapshot)
    if axis_lengths[0] <= 0.0:
        return 0.0, 0.0
    return float(axis_lengths[1] / axis_lengths[0]), float(axis_lengths[2] / axis_lengths[0])


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        return np.zeros_like(vector)
    return vector / norm


def compute_orientation_parameters(
    snapshot: np.ndarray,
    positions: Any | None = None,
    dt: float = 1.0,
    gravity: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute orientation parameters I1-G, V-G and V-I1.

    - I1-G: absolute dot product between smallest principal axis and gravity.
    - V-G: absolute dot product between mean flock velocity and gravity.
    - V-I1: absolute dot product between mean flock velocity and the smallest axis.
    """
    gravity_vec = np.array([0.0, 0.0, 1.0]) if gravity is None else _unit_vector(np.asarray(gravity, dtype=np.float64))
    axis_lengths, eigvecs = compute_principal_axes(snapshot)
    i1_axis = _unit_vector(eigvecs[:, 0])

    if positions is not None:
        # use centre-of-mass velocity for flock velocity
        com_vel = compute_com_velocities(positions, dt=dt)
        if com_vel.size > 0:
            mean_velocity = np.mean(com_vel, axis=0)
        else:
            mean_velocity = np.zeros(3, dtype=np.float64)
    else:
        mean_velocity = np.zeros(3, dtype=np.float64)

    v_dir = _unit_vector(mean_velocity)
    i1_g = float(abs(np.dot(i1_axis, gravity_vec)))
    v_g = float(abs(np.dot(v_dir, gravity_vec)))
    v_i1 = float(abs(np.dot(v_dir, i1_axis)))
    return i1_g, v_g, v_i1


def compute_balance_shift(snapshot: np.ndarray, v_dir: np.ndarray | None = None) -> float:
    """Compute balance shift along the direction of motion.

    Defined as the signed projection of (COM - geometric_center) onto the
    flock velocity direction, normalized by half the flock extent along that
    direction. Positive values indicate more mass towards the front.
    """
    # geometric centre: centroid of convex hull vertices when available
    if ConvexHull is not None:
        try:
            hull = ConvexHull(snapshot)
            geom_center = np.mean(snapshot[hull.vertices], axis=0)
        except Exception:
            geom_center = compute_centroid(snapshot)
    else:
        geom_center = compute_centroid(snapshot)

    com = compute_centroid(snapshot)

    if v_dir is None:
        try:
            _, eigvecs = compute_principal_axes(snapshot)
            v_dir = eigvecs[:, 2]
        except Exception:
            v_dir = np.array([1.0, 0.0, 0.0])
    v_dir = _unit_vector(np.asarray(v_dir, dtype=np.float64))
    s = float(np.dot(com - geom_center, v_dir))

    projections = np.dot(snapshot, v_dir)
    extent = float(np.max(projections) - np.min(projections))
    if extent <= 0.0:
        return 0.0
    # normalize by half-extent so that shift ~ +/-1 corresponds to a half-length
    return float(s / (extent / 2.0))


def compute_concavity(snapshot: np.ndarray) -> float:
    """Compute concavity as relative volume difference between convex hull
    and the alpha-shape (a-shape) representing the actual flock boundary.
    """
    convex_vol = compute_convex_hull_volume(snapshot)
    if convex_vol <= 0.0:
        return 0.0
    alpha_vol = compute_alpha_shape_volume(snapshot)
    concavity = (convex_vol - alpha_vol) / convex_vol
    return float(max(0.0, min(1.0, concavity)))


def read_positions_csv(path: Path) -> np.ndarray:
    """Read bird positions from a CSV file into shape (T, N, 3)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        expected = {"tick", "bird_id", "x", "y", "z"}
        header = {name.strip() for name in reader.fieldnames}
        if not expected.issubset(header):
            raise ValueError(
                f"CSV header must contain: {sorted(expected)}; got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                (
                    int(row["tick"]),
                    int(row["bird_id"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
            )

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    rows.sort(key=lambda item: (item[0], item[1]))
    ticks = sorted({tick for tick, *_ in rows})
    bird_ids = sorted({bird_id for _, bird_id, *_ in rows})
    tick_to_index = {tick: idx for idx, tick in enumerate(ticks)}
    bird_to_index = {bird_id: idx for idx, bird_id in enumerate(bird_ids)}

    positions = np.empty((len(ticks), len(bird_ids), 3), dtype=np.float64)
    positions.fill(np.nan)

    for tick, bird_id, x, y, z in rows:
        positions[tick_to_index[tick], bird_to_index[bird_id]] = (x, y, z)

    if np.isnan(positions).any():
        raise ValueError("CSV file has missing position values for some tick/bird combinations")

    return positions


def print_flock_metrics(metrics: dict[str, float]) -> None:
    """Print the flock metrics in a readable form."""
    print("Flock metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def compute_metrics_for_directory(
    folder: Path,
    output_csv: Path,
    dt: float = 1.0,
    recursive: bool = False,
) -> Path:
    """Read all CSV files from a directory, compute metrics, and write a summary CSV."""
    if not folder.is_dir():
        raise ValueError(f"Directory not found: {folder}")

    globber = folder.rglob if recursive else folder.glob
    csv_files = sorted(
        [path for path in globber("*.csv") if path.is_file() and path.resolve() != output_csv.resolve()]
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {folder}")

    fieldnames = [
        "source_file",
        "number_of_birds",
        "volume_m3",
        "density_r",
        "nnd_r1",
        "velocity_m_s",
        "concavity",
        "balance_shift",
        "thickness_I1",
        "I2_I1",
        "I3_I1",
        "I1_G",
        "V_G",
        "V_I1",
    ]

    rows = []
    for csv_path in csv_files:
        positions = read_positions_csv(csv_path)
        metrics = compute_flock_metrics(positions, dt=dt)
        metrics["source_file"] = str(csv_path.relative_to(folder))
        rows.append(metrics)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_csv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute flock metrics from position CSV or directory.")
    parser.add_argument("path", type=Path, help="Path to a CSV file or directory containing CSV files")
    parser.add_argument("--dt", type=float, default=1.0, help="Time interval between snapshots")
    parser.add_argument("--output", type=Path, default=Path("flock_metrics_summary.csv"), help="Output CSV path for directory batch processing")
    parser.add_argument("--recursive", action="store_true", help="Search CSV files recursively in subdirectories")
    args = parser.parse_args()

    if args.path.is_dir():
        output_csv = compute_metrics_for_directory(
            args.path,
            args.output,
            dt=args.dt,
            recursive=args.recursive,
        )
        print(f"Saved metrics summary to {output_csv}")
    else:
        positions = read_positions_csv(args.path)
        metrics = compute_flock_metrics(positions, dt=args.dt)
        print_flock_metrics(metrics)


def compute_flock_metrics(positions: Any, dt: float = 1.0) -> dict[str, float]:
    """Compute all available flock metrics from position records.

    Metrics that can be computed per snapshot are averaged across the available
    time series. Velocity is computed from the full sequence of positions.
    """
    arr = _normalize_positions(positions)
    if arr.shape[0] == 0:
        raise ValueError("positions must contain at least one snapshot")

    num_snapshots = arr.shape[0]
    # compute global flock velocity direction from COM trajectory
    com_vel = compute_com_velocities(arr, dt=dt)
    global_v_dir = None
    if com_vel.size > 0:
        gv = np.mean(com_vel, axis=0)
        global_v_dir = _unit_vector(gv)

    snapshot_metrics = []
    for snapshot in arr:
        i1_g, v_g, v_i1 = compute_orientation_parameters(snapshot, positions=arr, dt=dt)
        snapshot_metrics.append(
            {
                "volume_m3": compute_convex_hull_volume(snapshot),
                "density_r": compute_flock_density(snapshot),
                "nnd_r1": compute_nearest_neighbor_distance(snapshot),
                "concavity": compute_concavity(snapshot),
                "balance_shift": compute_balance_shift(snapshot, v_dir=global_v_dir),
                "thickness_I1": compute_thickness_I1(snapshot),
                "I2_I1": compute_aspect_ratios(snapshot)[0],
                "I3_I1": compute_aspect_ratios(snapshot)[1],
                "I1_G": i1_g,
                "V_G": v_g,
                "V_I1": v_i1,
            }
        )

    mean_metrics = {
        metric: float(np.mean([snap[metric] for snap in snapshot_metrics]))
        for metric in snapshot_metrics[0]
    }

    return {
        "number_of_birds": float(compute_number_of_birds(arr)),
        "volume_m3": mean_metrics["volume_m3"],
        "density_r": mean_metrics["density_r"],
        "nnd_r1": mean_metrics["nnd_r1"],
        "velocity_m_s": compute_com_mean_speed(arr, dt=dt),
        "concavity": mean_metrics["concavity"],
        "balance_shift": mean_metrics["balance_shift"],
        "thickness_I1": mean_metrics["thickness_I1"],
        "I2_I1": mean_metrics["I2_I1"],
        "I3_I1": mean_metrics["I3_I1"],
        "I1_G": mean_metrics["I1_G"],
        "V_G": mean_metrics["V_G"],
        "V_I1": mean_metrics["V_I1"],
    }


if __name__ == "__main__":
    main()
