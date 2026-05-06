import sys
import time
import argparse
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not installed. Run:  pip install onnxruntime")

import os

os.environ.setdefault("DISPLAY", ":0")  # harmless on pure Wayland

try:
    import vispy

    _BACKENDS = ["pyqt6", "pyqt5", "pyside6", "pyside2", "pyglet", "glfw"]
    _chosen = None
    for _b in _BACKENDS:
        try:
            vispy.use(_b)
            _chosen = _b
            break
        except Exception:
            continue
    if _chosen is None:
        print(
            "WARNING: Could not find PyQt5/6, PySide2/6, pyglet or glfw.\n"
            "Install one of them, e.g.:  pip install PyQt5\n"
            "Falling back to vispy auto-detection (may not open a window).",
            file=sys.stderr,
        )
    from vispy import app, scene
    from vispy.scene import visuals
except ImportError:
    sys.exit("vispy not installed. Run:\n" "  pip install vispy pyopengl PyQt5")


class FlockEnv:
    def __init__(
        self,
        num_birds: int = 100,
        world_size: float = 20.0,
        max_speed: float = 1.0,
        neighbor_radius: float = 3.0,
    ):
        self.num_birds = num_birds
        self.world_size = world_size
        self.max_speed = max_speed
        self.neighbor_radius = neighbor_radius
        self.neighbor_radius_sq = neighbor_radius**2

        self.reset()

    def reset(self):
        ws = self.world_size
        self.pos = np.random.uniform(0, ws, (self.num_birds, 3)).astype(np.float32)
        self.vel = np.random.uniform(-0.5, 0.5, (self.num_birds, 3)).astype(np.float32)
        # zero out z-component of velocity initially for gentler start
        self.vel[:, 2] *= 0.3

    def get_obs_batch(self) -> np.ndarray:
        N = self.num_birds
        obs = np.zeros((N, 6), dtype=np.float32)
        obs[:, 0:2] = self.vel[:, 0:2]  # self vel xy

        # pairwise relative positions (xy)
        # shape: (N, N, 2)
        rel = self.pos[np.newaxis, :, :2] - self.pos[:, np.newaxis, :2]  # (N,N,2)
        dist_sq = (rel**2).sum(axis=-1)  # (N,N)
        np.fill_diagonal(dist_sq, np.inf)
        mask = dist_sq < self.neighbor_radius_sq  # (N,N) bool

        counts = mask.sum(axis=1)  # (N,)

        # avg relative position
        rel_masked = rel * mask[:, :, np.newaxis]  # (N,N,2)
        avg_rel = rel_masked.sum(axis=1) / np.maximum(counts[:, np.newaxis], 1)

        # avg neighbor velocity
        vel_xy = self.vel[:, :2]  # (N,2)
        nbr_vel = (vel_xy[np.newaxis, :, :] * mask[:, :, np.newaxis]).sum(axis=1)
        avg_nbr_vel = nbr_vel / np.maximum(counts[:, np.newaxis], 1)

        obs[:, 2:4] = avg_rel
        obs[:, 4:6] = avg_nbr_vel

        # birds with no neighbours get zero for neighbour terms (already set)
        return obs

    def step(self, actions_xy: np.ndarray, dt: float = 1.0):
        """actions_xy: (N,2) delta vel from policy"""
        # apply action to XY velocity
        self.vel[:, 0:2] += np.clip(actions_xy, -1, 1) * 0.05 * dt

        # small random Z drift so flock undulates in 3-D
        self.vel[:, 2] += np.random.uniform(-0.02, 0.02, self.num_birds) * dt
        self.vel[:, 2] = np.clip(
            self.vel[:, 2], -self.max_speed * 0.4, self.max_speed * 0.4
        )

        # clamp speed
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        too_fast = (speed > self.max_speed).squeeze()
        self.vel[too_fast] /= speed[too_fast]
        self.vel[too_fast] *= self.max_speed

        # integrate positions with wrap
        self.pos += self.vel * dt
        self.pos %= self.world_size


# ─────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────


def speed_to_color(vel: np.ndarray, max_speed: float) -> np.ndarray:
    """Map speed to a colour gradient: deep blue → teal → gold."""
    speed = np.linalg.norm(vel, axis=1)  # (N,)
    t = np.clip(speed / max_speed, 0, 1)[:, None]  # (N,1)

    slow = np.array([0.15, 0.40, 0.85, 1.0])  # blue
    mid = np.array([0.10, 0.80, 0.70, 1.0])  # teal
    fast = np.array([1.00, 0.75, 0.10, 1.0])  # gold

    # two-stop gradient
    t2 = t * 2
    colors = np.where(
        t < 0.5,
        slow * (1 - t2) + mid * t2,
        mid * (1 - (t2 - 1)) + fast * (t2 - 1),
    )
    return colors.astype(np.float32)


# ─────────────────────────────────────────────
#  Main visualizer
# ─────────────────────────────────────────────


class StarlingViz:
    def __init__(
        self,
        onnx_path: str,
        num_birds: int = 100,
        world_size: float = 20.0,
        target_fps: float = 30.0,
    ):

        self.num_birds = num_birds
        self.world_size = world_size
        self.target_fps = target_fps
        self.dt_scale = 1.0
        self.paused = False
        self.step_count = 0

        # ── ONNX session ──────────────────────────────────────────────
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(onnx_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Try batched first; fall back to per-bird if model has fixed batch=1
        self._batched_inference = True
        print(f"Loaded ONNX model: {onnx_path}")

        # ── Environment ───────────────────────────────────────────────
        self.env = FlockEnv(
            num_birds=num_birds,
            world_size=world_size,
            max_speed=1.0,
            neighbor_radius=3.0,
        )

        # ── Vispy canvas + 3-D scene ───────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title="Starling Flock — ONNX",
            size=(1280, 800),
            bgcolor="#0a0a12",
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=world_size * 2.2,
            center=(world_size / 2,) * 3,
            elevation=25,
            azimuth=45,
        )

        # ── Spheres as Markers ─────────────────────────────────────────
        pos0 = self.env.pos.copy()
        col0 = speed_to_color(self.env.vel, self.env.max_speed)
        self.scatter = visuals.Markers(
            spherical=True,
            scaling=True,
            alpha=1.0,
        )
        self.scatter.set_data(
            pos0,
            face_color=col0,
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self.view.add(self.scatter)

        # ── Wire-frame world box ───────────────────────────────────────
        ws = world_size
        box_verts = np.array(
            [
                [0, 0, 0],
                [ws, 0, 0],
                [ws, ws, 0],
                [0, ws, 0],
                [0, 0, 0],
                [0, 0, ws],
                [ws, 0, ws],
                [ws, ws, ws],
                [0, ws, ws],
                [0, 0, ws],
            ],
            dtype=np.float32,
        )
        box_line = scene.visuals.Line(
            pos=box_verts,
            color=(0.3, 0.3, 0.5, 0.4),
            width=1,
            connect="strip",
            parent=self.view.scene,
        )
        # four vertical pillars
        for x, y in [(0, 0), (ws, 0), (ws, ws), (0, ws)]:
            scene.visuals.Line(
                pos=np.array([[x, y, 0], [x, y, ws]], dtype=np.float32),
                color=(0.3, 0.3, 0.5, 0.4),
                width=1,
                parent=self.view.scene,
            )
        # remaining horizontal edges at top
        for (x1, y1), (x2, y2) in [
            ((0, 0), (ws, 0)),
            ((ws, 0), (ws, ws)),
            ((ws, ws), (0, ws)),
            ((0, ws), (0, 0)),
        ]:
            scene.visuals.Line(
                pos=np.array([[x1, y1, ws], [x2, y2, ws]], dtype=np.float32),
                color=(0.3, 0.3, 0.5, 0.4),
                width=1,
                parent=self.view.scene,
            )

        # ── HUD text ──────────────────────────────────────────────────
        self.hud = scene.visuals.Text(
            "",
            color="white",
            font_size=10,
            anchor_x="left",
            anchor_y="top",
            parent=self.canvas.scene,
        )
        self.hud.pos = (12, 12, 0)

        # ── Timer ─────────────────────────────────────────────────────
        self._last_t = time.perf_counter()
        self._fps_acc = 0.0
        self._fps_n = 0
        self._fps_disp = 0.0
        self.timer = app.Timer(
            interval=1.0 / target_fps,
            connect=self._tick,
            start=True,
        )

    def _run_per_bird(self, obs: np.ndarray) -> np.ndarray:
        """Run inference one bird at a time for fixed-batch-size models."""
        results = []
        for i in range(len(obs)):
            out = self.session.run(
                [self.output_name],
                {self.input_name: obs[i : i + 1]},  # shape (1,6)
            )[
                0
            ]  # shape (1,2)
            results.append(out[0])
        return np.stack(results, axis=0)  # (N,2)

    # ── Simulation tick ───────────────────────────────────────────────

    def _tick(self, event):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now

        # FPS tracking
        self._fps_acc += 1.0 / max(dt, 1e-6)
        self._fps_n += 1
        if self._fps_n >= 15:
            self._fps_disp = self._fps_acc / self._fps_n
            self._fps_acc = 0.0
            self._fps_n = 0

        if self.paused:
            self._update_hud()
            return

        # ── inference ──
        obs = self.env.get_obs_batch()  # (N,6)
        # Model was exported with dummy_input shape (1,6), so batch dim is
        # fixed to 1. Run inference per-bird and stack results.
        if self._batched_inference:
            try:
                raw = self.session.run([self.output_name], {self.input_name: obs})[
                    0
                ]  # (N,2)
            except Exception:
                self._batched_inference = False
                raw = self._run_per_bird(obs)
        else:
            raw = self._run_per_bird(obs)
        self.env.step(raw, dt=self.dt_scale)
        self.step_count += 1

        # ── update visuals ──
        colors = speed_to_color(self.env.vel, self.env.max_speed)
        self.scatter.set_data(
            self.env.pos.copy(),
            face_color=colors,
            edge_color=None,
            size=0.45,
            edge_width=0,
        )
        self._update_hud()

    def _update_hud(self):
        state = "PAUSED" if self.paused else "RUNNING"
        self.hud.text = (
            f"Birds: {self.num_birds}   Step: {self.step_count}   "
            f"FPS: {self._fps_disp:.1f}   Speed: {self.dt_scale:.2f}x   "
            f"[{state}]\n"
            f"[Space] pause  [R] reset cam  [+/-] speed  [Q] quit"
        )

    # ── Key bindings ──────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.key.name if event.key else ""
        if k in ("Q", "Escape"):
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

    def run(self):
        app.run()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-D Starling Flock Visualization (ONNX)",
        epilog=(
            "Wayland/Arch tip: if no window appears, install PyQt5 or PyQt6:\n"
            "  pip install PyQt5\n"
            "Then optionally force it:  --backend pyqt5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "onnx",
        nargs="?",
        default="starling_policy.onnx",
        help="Path to exported .onnx policy file (default: starling_policy.onnx)",
    )
    parser.add_argument("--birds", type=int, default=100, help="Number of birds")
    parser.add_argument("--world", type=float, default=20.0, help="World size")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument(
        "--backend",
        default=None,
        help="Force vispy backend, e.g. pyqt5 pyqt6 pyside6 pyglet glfw",
    )
    args = parser.parse_args()

    # Allow command-line override of the backend selected at import time
    if args.backend:
        try:
            vispy.use(args.backend)
            print(f"vispy backend forced to: {args.backend}")
        except Exception as e:
            print(
                f"WARNING: could not set backend '{args.backend}': {e}", file=sys.stderr
            )

    viz = StarlingViz(
        onnx_path=args.onnx,
        num_birds=args.birds,
        world_size=args.world,
        target_fps=args.fps,
    )
    viz.run()
