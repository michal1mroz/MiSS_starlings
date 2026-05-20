from __future__ import annotations

import csv
import sys
import time
import argparse
import datetime
import numpy as np
from pathlib import Path

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
from starling_env import StarlingFlockEnv


def speed_to_color(vel: np.ndarray, max_speed: float) -> np.ndarray:
    """Blue → teal → gold gradient keyed on speed."""
    t = np.clip(np.linalg.norm(vel, axis=1) / max_speed, 0, 1)[:, None]
    slow = np.array([0.15, 0.40, 0.85, 1.0])
    mid = np.array([0.10, 0.80, 0.70, 1.0])
    fast = np.array([1.00, 0.75, 0.10, 1.0])
    t2 = t * 2
    return np.where(
        t < 0.5, slow * (1 - t2) + mid * t2, mid * (1 - (t2 - 1)) + fast * (t2 - 1)
    ).astype(np.float32)


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


class StarlingViz:
    def __init__(
        self,
        onnx_path: str,
        num_birds: int = 100,
        world_size: float = 40.0,
        generation_range: float = 20.0,
        target_fps: float = 30.0,
        epsilon: float = 0.10,
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

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(onnx_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Loaded: {onnx_path}")

        self.env = StarlingFlockEnv(
            num_birds=num_birds,
            world_size=world_size,
            max_speed=1.0,
            neighbor_sigma=sigma,
            spawn_range=self.generation_range,
            device="cpu",
        )
        self.env.reset()

        if log_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/flock_log_{ts}.csv"
        self.logger = FlockLogger(log_path, print_every=log_print_every)
        self.logging_enabled = True

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

        self.scatter = visuals.Markers(spherical=True, scaling=True, alpha=1.0)
        self.scatter.set_data(
            self.env.pos_np,
            face_color=speed_to_color(self.env.vel_np, self.env.max_speed),
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self.view.add(self.scatter)

        self._draw_box(ws, color=(0.3, 0.3, 0.5, 0.3))

        #        if self.generation_range < ws:
        #            half = ws / 2.0
        #            gr = self.generation_range
        #            offset = half - gr / 2.0
        #            self._draw_box(gr, color=(0.4, 0.7, 0.4, 0.5), offset=offset)

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

    def _draw_box(self, size: float, color: tuple, offset: float = 0.0) -> None:
        """Draw a wireframe cube of `size` starting at `offset` on each axis."""
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
                        [self.output_name], {self.input_name: obs_np[i : i + 1]}
                    )[0][0]
                    for i in range(self.num_birds)
                ]
            )

        if self.epsilon > 0:
            N = self.num_birds
            mask = np.random.rand(N) < self.epsilon
            noise = np.random.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
            actions = np.where(mask[:, None], noise, actions)

        self.env.step(actions)
        self.step_count += 1

        stats = self.env.compute_stats()
        self._last_stats = stats

        if self.logging_enabled and self.step_count % self.log_every == 0:
            self.logger.log(self.step_count, stats)

        colors = speed_to_color(self.env.vel_np, self.env.max_speed)
        self.scatter.set_data(
            self.env.pos_np,
            face_color=colors,
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self._update_hud()

    def _update_hud(self):
        s = self._last_stats
        state = "PAUSED" if self.paused else "RUNNING"
        log_s = "ON " if self.logging_enabled else "OFF"

        self.hud_stats.text = (
            f"[{state}] Step: {self.step_count:6d} "
            # f"FPS: {self._fps_disp:5.1f}   "
            f"Speed×: {self.dt_scale:.2f} "
            f"Eps: {self.epsilon:.2f} "
            f"Log: {log_s}\n"
            f"mean speed: {s.get('mean_speed',    0):.4f}\n"
            f"mean NND: {s.get('mean_nnd',      0):.4f}\n"
            f"polarity: {s.get('global_polarity',0):.4f}\n"
            f"flock radius: {s.get('flock_radius',  0):.4f}\n"
            f"mean reward: {s.get('mean_reward',   0):.4f}"
        )

        self.hud_ctrl.text = (
            "[Space] pause   [R] reset cam   [+/-] sim speed   "
            "[E/D] epsilon [+/-] 0.05   [L] toggle log   [Q] quit"
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
    parser.add_argument("--epsilon", type=float, default=0.10)
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
