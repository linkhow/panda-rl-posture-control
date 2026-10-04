"""Quintic timing of a straight path, expressed in world metres and seconds."""
from dataclasses import dataclass

import numpy as np


@dataclass
class QuinticLine:
    start: np.ndarray
    goal: np.ndarray
    duration: float

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float).copy()
        self.goal = np.asarray(self.goal, dtype=float).copy()
        if (self.start.shape != (3,) or self.goal.shape != (3,)
                or not np.isfinite([*self.start, *self.goal, self.duration]).all()
                or self.duration <= 0):
            raise ValueError("Expected finite 3D endpoints and positive duration")

    def sample(self, t: float):
        """Before t=0 hold start; at/after T hold goal, with zero v and a."""
        if not np.isfinite(t):
            raise ValueError("Non-finite reference time")
        u = float(np.clip(t / self.duration, 0.0, 1.0))
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / self.duration
        dds = (60 * u - 180 * u**2 + 120 * u**3) / self.duration**2
        delta = self.goal - self.start
        return self.start + s * delta, ds * delta, dds * delta
