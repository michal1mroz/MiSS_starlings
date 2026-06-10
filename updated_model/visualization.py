"""
visualization.py — Starling Flock Visualiser with Leader Birds

Leader birds are assigned randomly at reset. They:
  - Have elevated neighbour weight (LEADER_WEIGHT) so the flock steers toward them
  - Take random actions with probability LEADER_EPSILON (vs FOLLOWER_EPSILON for others)
  - Are rendered in a distinct colour with a translucent sphere halo

All tunable knobs live in the FLOCK CONFIG block below.
"""

from __future__ import annotations

import csv
import sys
import time
import argparse
import datetime
import numpy as np
from pathlib import Path


NUM_LEADERS: int = 1
LEADER_WEIGHT: float = 20.0
LEADER_EPSILON: float = 0.50

FOLLOWER_EPSILON: float = 0.1
LEADER_COLOR: tuple = (1.00, 0.30, 0.10, 1.0)

LEADER_HALO_COLOR: tuple = (1.00, 0.30, 0.10, 0.18)
LEADER_HALO_RADIUS: float = 1.2


try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not installed.  pip install onnxruntime")

import os

os.environ.setdefault("DISPLAY", ":0")

try:
    import vispy

    for _b in ["pyqt6", "pyqt5", "pyside6", "pyside2", "pyglet", "glfw"]:
        try:
            vispy.use(_b)
            break
        except Exception:
            continue
    from vispy import app, scene
    from vispy.scene import visuals
except ImportError:
    sys.exit("vispy not installed.  pip install vispy pyopengl PyQt5")

sys.path.insert(0, os.path.dirname(__file__))
from starling_env import StarlingFlockEnv, StarlingLeaderEnv


def speed_to_color(vel: np.ndarray, max_speed: float) -> np.ndarray:
    t = np.clip(np.linalg.norm(vel, axis=1) / max_speed, 0, 1)[:, None]
    slow = np.array([0.15, 0.40, 0.85, 1.0])
    mid = np.array([0.10, 0.80, 0.70, 1.0])
    fast = np.array([1.00, 0.75, 0.10, 1.0])
    t2 = t * 2
    return np.where(
        t < 0.5, slow * (1 - t2) + mid * t2, mid * (1 - (t2 - 1)) + fast * (t2 - 1)
    ).astype(np.float32)


def build_halo_sphere(radius: float = 1.0, rows: int = 12, cols: int = 16) -> np.ndarray:
    verts = []

    for r in range(1, rows):
        phi = np.pi * r / rows
        z = radius * np.cos(phi)
        rr = radius * np.sin(phi)
        theta = np.linspace(0, 2 * np.pi, cols + 1)
        ring = np.column_stack([rr * np.cos(theta), rr * np.sin(theta),
                                 np.full(cols + 1, z)])
        verts.append(ring)
        verts.append(np.full((1, 3), np.nan))

    for c in range(cols):
        theta = 2 * np.pi * c / cols
        phi = np.linspace(0, np.pi, rows + 1)
        line = np.column_stack([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ])
        verts.append(line)
        verts.append(np.full((1, 3), np.nan))

    return np.concatenate(verts, axis=0).astype(np.float32)


def _patch_leader_weights(env: StarlingFlockEnv, leader_ids: list[int]) -> None:
    import torch

    _original_compute_weights = env.__class__._compute_weights

    def patched_compute_weights(self):
        w = _original_compute_weights(self)
        import torch as _torch
        bonus = _torch.zeros(w.shape[0], device=w.device)
        for lid in leader_ids:
            bonus[lid] = LEADER_WEIGHT
        w = w + bonus.unsqueeze(0)
        w.fill_diagonal_(0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return w

    import types
    env._compute_weights = types.MethodType(patched_compute_weights, env)


class FlockLogger:
    FIELDS = [
        "step",
        "wall_time",
        "mean_speed",
        "mean_nnd",
        "global_polarity",
        "flock_radius",
        "mean_reward",
    ]

    def __init__(self, log_path: str | Path, print_every: int = 60):
        self.log_path = Path(log_path)
        self.print_every = print_every
        self._file = self.log_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._t0 = time.perf_counter()
        print(f"[Logger] writing to {self.log_path.resolve()}")

    def log(self, step: int, stats: dict[str, float]) -> None:
        row = {
            "step": step,
            "wall_time": round(time.perf_counter() - self._t0, 3),
            **{k: round(v, 5) for k, v in stats.items()},
        }
        self._writer.writerow(row)

        if self.print_every > 0 and step % self.print_every == 0:
            print(
                f"[step {step:6d}] "
                f"spd={stats['mean_speed']:.3f}  "
                f"nnd={stats['mean_nnd']:.3f}  "
                f"pol={stats['global_polarity']:.3f}  "
                f"rad={stats['flock_radius']:.2f}  "
                f"rew={stats['mean_reward']:.4f}"
            )

    def close(self) -> None:
        self._file.close()
        print(f"[Logger] closed {self.log_path}")


class LeaderHalos:
    def __init__(self, view, num_leaders: int, radius: float, color: tuple):
        self._template = build_halo_sphere(radius)
        self._visuals: list = []
        for _ in range(num_leaders):
            v = scene.visuals.Line(
                pos=self._template,
                color=color,
                connect="strip",
                width=1,
                parent=view.scene,
            )
            self._visuals.append(v)

    def update(self, positions: np.ndarray) -> None:
        """Move each sphere to the corresponding leader position."""
        for v, pos in zip(self._visuals, positions):
            moved = self._template + pos[None, :]
            v.set_data(pos=moved)

    def set_visible(self, visible: bool) -> None:
        for v in self._visuals:
            v.visible = visible


class StarlingViz:
    def __init__(
        self,
        onnx_path: str,
        num_birds: int = 100,
        world_size: float = 40.0,
        generation_range: float = 20.0,
        target_fps: float = 30.0,
        epsilon: float = FOLLOWER_EPSILON,
        sigma: float = 3.0,
        log_path: str | None = None,
        log_every: int = 1,
        log_print_every: int = 60,
    ):
        self.num_birds = num_birds
        self.world_size = world_size
        self.generation_range = min(generation_range, world_size)
        self.target_fps = target_fps
        self.epsilon = epsilon
        self.dt_scale = 1.0
        self.paused = True
        self.step_count = 0
        self.log_every = log_every
        self._last_stats: dict[str, float] = {}

        # ---- ONNX session ----
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(onnx_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Loaded: {onnx_path}")

        # ---- Environment ----
        self.env = StarlingFlockEnv(
            num_birds=num_birds,
            world_size=world_size,
            max_speed=1.0,
            neighbor_sigma=sigma,
            spawn_range=self.generation_range,
            device="cpu",
        )

        # elect leaders and patch weight computation
        self.leader_ids: list[int] = sorted(
            np.random.choice(num_birds, size=min(NUM_LEADERS, num_birds), replace=False).tolist()
        )
        print(f"[Leaders] bird indices: {self.leader_ids}  "
              f"(weight bonus={LEADER_WEIGHT}, ε={LEADER_EPSILON})")
        _patch_leader_weights(self.env, self.leader_ids)

        self.env.reset()

        # ---- Logger ----
        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/flock_log_{ts}.csv"
        self.logger = FlockLogger(log_path, print_every=log_print_every)
        self.logging_enabled = True

        # ---- Canvas ----
        self.canvas = scene.SceneCanvas(
            title="Starling Flock — ONNX",
            size=(1280, 800),
            bgcolor="#0a0a12",
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        self.view = self.canvas.central_widget.add_view()
        ws = world_size
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=ws * 2.2,
            center=(ws / 2,) * 3,
            elevation=25,
            azimuth=45,
        )

        # ---- Bird scatter ----
        self._colors = self._build_colors()
        self.scatter = visuals.Markers(spherical=True, scaling=True, alpha=1.0)
        self.scatter.set_data(
            self.env.pos_np,
            face_color=self._colors,
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self.view.add(self.scatter)

        # ---- Leader halos ----
        self.halos = LeaderHalos(
            self.view,
            num_leaders=len(self.leader_ids),
            radius=LEADER_HALO_RADIUS,
            color=LEADER_HALO_COLOR,
        )
        self.halos.update(self.env.pos_np[self.leader_ids])

        # ---- World box ----
        self._draw_box(ws, color=(0.3, 0.3, 0.5, 0.3))

        # ---- HUD ----
        self.hud_stats = scene.visuals.Text(
            "",
            color="white",
            font_size=10,
            anchor_x="left",
            anchor_y="top",
            parent=self.canvas.scene,
        )
        self.hud_stats.pos = (12, 115, 0)

        self.hud_ctrl = scene.visuals.Text(
            "",
            color=(0.7, 0.7, 0.7, 1.0),
            font_size=10,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.hud_ctrl.pos = (12, self.canvas.size[1] - 24, 0)

        self._last_t = time.perf_counter()
        self._fps_acc = 0.0
        self._fps_n = 0
        self._fps_disp = 0.0
        self.timer = app.Timer(
            interval=1.0 / target_fps,
            connect=self._tick,
            start=True,
        )

    def _build_colors(self) -> np.ndarray:
        """Speed-based colours for all birds, leaders overridden to LEADER_COLOR."""
        colors = speed_to_color(self.env.vel_np, self.env.max_speed)
        lc = np.array(LEADER_COLOR, dtype=np.float32)
        for lid in self.leader_ids:
            colors[lid] = lc
        return colors

    def _update_colors(self) -> np.ndarray:
        colors = speed_to_color(self.env.vel_np, self.env.max_speed)
        lc = np.array(LEADER_COLOR, dtype=np.float32)
        for lid in self.leader_ids:
            colors[lid] = lc
        return colors
    

    def _leader_obs(self, leader_ids: list[int]) -> np.ndarray:
        """
        Build leader-policy observations matching StarlingLeaderEnv:
        [vel_x, vel_y, vel_z, pos_x, pos_y, pos_z, dist_x, dist_y, dist_z]
        with position centered in [-world/2, world/2].
        """
        half = self.world_size / 2.0

        vel = self.env.vel_np[leader_ids] / self.env.max_speed
        vel = np.clip(vel, -1.0, 1.0)

        centered_pos = self.env.pos_np[leader_ids] - half
        pos = centered_pos / half
        pos = np.clip(pos, -1.0, 1.0)

        dist_to_walls = (half - np.abs(centered_pos)) / half
        dist_to_walls = np.clip(dist_to_walls, 0.0, 1.0)

        return np.concatenate([vel, pos, dist_to_walls], axis=1).astype(np.float32)
    
    def _draw_box(self, size: float, color: tuple, offset: float = 0.0) -> None:
        o = offset
        s = size
        kw = dict(color=color, width=1, parent=self.view.scene)

        base = np.array(
            [
                [o, o, o],
                [o + s, o, o],
                [o + s, o + s, o],
                [o, o + s, o],
                [o, o, o],
                [o, o, o + s],
                [o + s, o, o + s],
                [o + s, o + s, o + s],
                [o, o + s, o + s],
                [o, o, o + s],
            ],
            dtype=np.float32,
        )
        scene.visuals.Line(pos=base, connect="strip", **kw)

        for x, y in [(o + s, o), (o + s, o + s), (o, o + s)]:
            scene.visuals.Line(
                pos=np.array([[x, y, o], [x, y, o + s]], dtype=np.float32), **kw
            )

        for (x1, y1), (x2, y2) in [
            ((o, o), (o + s, o)),
            ((o + s, o), (o + s, o + s)),
            ((o + s, o + s), (o, o + s)),
            ((o, o + s), (o, o)),
        ]:
            scene.visuals.Line(
                pos=np.array([[x1, y1, o + s], [x2, y2, o + s]], dtype=np.float32), **kw
            )

    def _tick(self, _event):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now

        self._fps_acc += 1.0 / max(dt, 1e-6)
        self._fps_n += 1
        if self._fps_n >= 15:
            self._fps_disp = self._fps_acc / self._fps_n
            self._fps_acc = 0.0
            self._fps_n = 0

        if self.paused:
            self._update_hud()
            return

        obs_np = self.env.get_obs_tensor().numpy()

        try:
            actions = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        except Exception:
            actions = np.stack(
                [
                    self.session.run(
                        [self.output_name], {self.input_name: obs_np[i: i + 1]}
                    )[0][0]
                    for i in range(self.num_birds)
                ]
            )

        N = self.num_birds
        epsilons = np.full(N, self.epsilon, dtype=np.float32)
        for lid in self.leader_ids:
            epsilons[lid] = LEADER_EPSILON

        mask = np.random.rand(N) < epsilons
        noise = np.random.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
        actions = np.where(mask[:, None], noise, actions)

        self.env.step(actions)
        self.step_count += 1

        stats = self.env.compute_stats()
        self._last_stats = stats

        if self.logging_enabled and self.step_count % self.log_every == 0:
            self.logger.log(self.step_count, stats)

        colors = self._update_colors()
        self.scatter.set_data(
            self.env.pos_np,
            face_color=colors,
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self.halos.update(self.env.pos_np[self.leader_ids])

        self._update_hud()

    def _update_hud(self):
        s = self._last_stats
        state = "PAUSED" if self.paused else "RUNNING"
        log_s = "ON " if self.logging_enabled else "OFF"

        leader_str = ",".join(str(i) for i in self.leader_ids)
        self.hud_stats.text = (
            f"[{state}] Step: {self.step_count:6d}  "
            f"Speed×: {self.dt_scale:.2f}  "
            f"Eps(f/l): {self.epsilon:.2f}/{LEADER_EPSILON:.2f}  "
            f"Log: {log_s}\n"
            f"Leaders: [{leader_str}]  weight+{LEADER_WEIGHT:.0f}\n"
            f"mean speed:   {s.get('mean_speed',    0):.4f}\n"
            f"mean NND:     {s.get('mean_nnd',      0):.4f}\n"
            f"polarity:     {s.get('global_polarity',0):.4f}\n"
            f"flock radius: {s.get('flock_radius',  0):.4f}\n"
            f"mean reward:  {s.get('mean_reward',   0):.4f}"
        )

        self.hud_ctrl.text = (
            "[Space] pause   [R] reset cam   [+/-] sim speed   "
            "[E/D] follower-ε ±0.05   [L] toggle log   [N] new leaders   [Q] quit"
        )

    def _on_key(self, event):
        k = event.key.name if event.key else ""
        if k in ("Q", "Escape"):
            self.logger.close()
            self.timer.stop()
            app.quit()
        elif k == "Space":
            self.paused = not self.paused
        elif k == "R":
            ws = self.world_size
            self.view.camera.center = (ws / 2, ws / 2, ws / 2)
            self.view.camera.distance = ws * 2.2
            self.view.camera.elevation = 25
            self.view.camera.azimuth = 45
        elif k in ("+", "Equal"):
            self.dt_scale = min(self.dt_scale * 1.25, 8.0)
        elif k in ("-", "Minus"):
            self.dt_scale = max(self.dt_scale / 1.25, 0.1)
        elif k == "E":
            self.epsilon = min(self.epsilon + 0.05, 1.0)
        elif k == "D":
            self.epsilon = max(self.epsilon - 0.05, 0.0)
        elif k == "L":
            self.logging_enabled = not self.logging_enabled
            state = "enabled" if self.logging_enabled else "disabled"
            print(f"[Logger] {state}")
        elif k == "N":
            # Re-elect leaders at runtime
            self.leader_ids = sorted(
                np.random.choice(
                    self.num_birds,
                    size=min(NUM_LEADERS, self.num_birds),
                    replace=False,
                ).tolist()
            )
            _patch_leader_weights(self.env, self.leader_ids)
            print(f"[Leaders] re-elected: {self.leader_ids}")

    def run(self):
        app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-D Starling Flock Visualiser")
    parser.add_argument("onnx", nargs="?", default="starling_policy.onnx")
    parser.add_argument("--birds", type=int, default=100)
    parser.add_argument(
        "--world", type=float, default=40.0, help="Total world size (default 40.0)"
    )
    parser.add_argument(
        "--gen-range",
        type=float,
        default=20.0,
        help="Spawn region side length centred in world (default 20.0)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=FOLLOWER_EPSILON,
        help=f"Follower random-action probability (default {FOLLOWER_EPSILON})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Neighbourhood sigma, should match training (default 3.0)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="CSV log file path (default: auto timestamped)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Write a log row every N steps (default 1)",
    )
    parser.add_argument(
        "--log-print-every",
        type=int,
        default=60,
        help="Print summary every N steps, 0=off (default 60)",
    )
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    if args.backend:
        try:
            vispy.use(args.backend)
        except Exception as e:
            print(f"WARNING: backend '{args.backend}': {e}")

    StarlingViz(
        onnx_path=args.onnx,
        num_birds=args.birds,
        world_size=args.world,
        generation_range=args.gen_range,
        target_fps=args.fps,
        epsilon=args.epsilon,
        sigma=args.sigma,
        log_path=args.log,
        log_every=args.log_every,
        log_print_every=args.log_print_every,
    ).run()
