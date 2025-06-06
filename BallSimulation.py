import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from EdgeDetection import CannyEdgeDetector
from ShapeDetector import ShapeDetector

class ODESolver:
    def euler_step(self, state: np.ndarray, derivatives_func: Callable, dt: float) -> np.ndarray:
        return state + dt * derivatives_func(state)

    def rk4_step(self, state: np.ndarray, derivatives_func: Callable, dt: float) -> np.ndarray:
        k1 = derivatives_func(state)
        k2 = derivatives_func(state + dt * k1 / 2)
        k3 = derivatives_func(state + dt * k2 / 2)
        k4 = derivatives_func(state + dt * k3)
        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self, initial_state: np.ndarray,
              derivatives_func: Callable,
              dt: float,
              max_steps: int,
              method: str = 'rk4',
              stop_condition: Callable = None) -> np.ndarray:
        step_func = self.rk4_step if method == 'rk4' else self.euler_step
        states = np.zeros((max_steps, len(initial_state)))
        states[0] = initial_state

        step = 1
        while step < max_steps:
            states[step] = step_func(states[step - 1], derivatives_func, dt)
            if stop_condition and stop_condition(states[step]):
                return states[:step + 1]
            step += 1
        return states[:step]

@dataclass
class PhysicsParams:
    g: float = 9.81
    k: float = 0.0

class ProjectilePhysics:
    def __init__(self, params: PhysicsParams = PhysicsParams()):
        self.params = params
        self.solver = ODESolver()

    def get_derivatives(self, state: np.ndarray) -> np.ndarray:
        _, _, vx, vy = state
        return np.array([vx, vy, 0, -self.params.g])

    def simulate_trajectory(self,
                            initial_state: np.ndarray,
                            target_x: float,
                            method: str = 'rk4',
                            dt: float = 0.01,
                            max_steps: int = 1000) -> np.ndarray:
        def stop_condition(state):
            return state[1] < 0 or state[0] > target_x + 50

        return self.solver.solve(
            initial_state=initial_state,
            derivatives_func=self.get_derivatives,
            dt=dt,
            max_steps=max_steps,
            method=method,
            stop_condition=stop_condition
        )

class TargetHitter:
    def __init__(self, physics: ProjectilePhysics):
        self.physics = physics

    def find_initial_velocity(self, start_pos: Tuple[float, float],
                              target_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        g = self.physics.params.g

        best_error = float('inf')
        best_velocity = None

        base_angle = np.arctan2(dy, dx)

        for angle_offset in np.linspace(-np.pi / 4, np.pi / 4, 40):
            angle = base_angle + angle_offset

            if not (0 < angle < np.pi / 2):
                continue

            try:
                v0 = np.sqrt((g * dx ** 2) / (2 * np.cos(angle) ** 2 * (dx * np.tan(angle) - dy)))

                if not (20 < v0 < 200):
                    continue

                vx = v0 * np.cos(angle)
                vy = v0 * np.sin(angle)

                initial_state = np.array([start_pos[0], start_pos[1], vx, vy])
                trajectory = self.physics.simulate_trajectory(initial_state, target_pos[0])

                dists = np.sqrt((trajectory[:, 0] - target_pos[0]) ** 2 +
                                (trajectory[:, 1] - target_pos[1]) ** 2)
                min_dist = np.min(dists)

                if min_dist < best_error:
                    best_error = min_dist
                    best_velocity = (vx, vy)

                    if min_dist < 10:
                        return best_velocity

            except (RuntimeWarning, ValueError):
                continue

        return best_velocity

class BallThrowingSimulation:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.edge_detector = CannyEdgeDetector(low_threshold=0.05, high_threshold=0.2)
        self.shape_detector = ShapeDetector(min_area=100, epsilon_factor=0.01, circle_threshold=0.8)

        self.physics = ProjectilePhysics()
        self.target_hitter = TargetHitter(self.physics)

        self.height = self.image.shape[0]
        self.start_pos = (0, 0)

        self.process_targets()
        self.setup_visualization()

    def process_targets(self):
        edges = self.edge_detector.detect(self.image)
        image_shapes = self.shape_detector.detect_shapes(edges)

        self.shapes = [(x, self.height - y, r) for x, y, r in image_shapes]
        self.shapes.sort(key=lambda c: c[0])

        self.trajectories = []
        print("\nCalculating trajectories:")

        for i, (x, y, r) in enumerate(self.shapes):
            print(f"\nTarget {i + 1} at ({x}, {y})")

            velocity = self.target_hitter.find_initial_velocity(self.start_pos, (x, y))
            if velocity:
                vx, vy = velocity
                print(f"Initial velocity: ({vx:.2f}, {vy:.2f}) m/s")

                initial_state = np.array([self.start_pos[0], self.start_pos[1], vx, vy])
                trajectory = self.physics.simulate_trajectory(initial_state, x)
                self.trajectories.append(trajectory)
            else:
                print(f"Could not find valid trajectory for target {i + 1}")

    def setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        img_display = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_display = np.flipud(img_display)
        self.ax.imshow(img_display, origin='lower')

        self.current_idx = 0
        self.frame_count = 0

        self.line, = self.ax.plot([], [], 'b-', alpha=0.7, label='Trajectory')
        self.ball, = self.ax.plot([], [], 'ro', markersize=8, label='Ball')

    def animate(self):
        for x, y, r in self.shapes:
            circle = plt.Circle((x, y), r, color='red', fill=False)
            self.ax.add_patch(circle)

        self.ax.plot(self.start_pos[0], self.start_pos[1], 'go',
                     markersize=10, label='Launch position')

        def init():
            self.line.set_data([], [])
            self.ball.set_data([], [])
            return [self.line, self.ball]

        def update(frame):
            if self.current_idx >= len(self.trajectories):
                return [self.line, self.ball]

            trajectory = self.trajectories[self.current_idx]
            progress = min(1.0, self.frame_count / 30)
            idx = int(progress * len(trajectory))

            if idx > 0:
                self.line.set_data(trajectory[:idx, 0], trajectory[:idx, 1])
                self.ball.set_data([trajectory[idx - 1, 0]], [trajectory[idx - 1, 1]])

            self.frame_count += 1
            if progress >= 1.0:
                self.current_idx += 1
                self.frame_count = 0
                self.line.set_data([], [])
                self.ball.set_data([], [])

            return [self.line, self.ball]

        anim = FuncAnimation(self.fig, update, init_func=init,
                             frames=None, interval=20,
                             save_count=len(self.trajectories) * 30)

        plt.legend()
        plt.title("Ball Throwing Simulation")
        plt.show()

def main():
    sim = BallThrowingSimulation("Untitled design.png")
    sim.animate()

if __name__ == "__main__":
    main()