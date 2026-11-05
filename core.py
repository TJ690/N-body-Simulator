"""Core of the N-body simulator: bodies, gravity, integrators and the simulation loop.

Sections
    1. Body            data structure for one gravitating body
    2. Physics         all-pairs forces, energy diagnostics, time integrators
    3. Simulation      owns the state and advances time

No plotting code lives here, so this module runs headless with only NumPy.

The physics in one line (Newtonian gravity with Plummer softening):

    a_i = G * sum over j != i of  m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

"Direct all-pairs" means we evaluate that sum for every one of the N bodies
against every other body, so the cost per force evaluation grows as O(N^2).
That is simple and exact, and perfectly fine for small N.

Array shapes used throughout (N = number of bodies):
    pos, vel, acc   (N, 3)   one row per body, columns are x, y, z
    mass            (N,)

Every integrator has the same signature:

    step(pos, vel, acc, dt, accel_fn) -> (new_pos, new_vel, new_acc)

where `acc` is the acceleration at `pos` and `new_acc` is the acceleration at
`new_pos` (so the next step can reuse it instead of recomputing). `accel_fn(pos)`
returns accelerations for any set of positions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ==========================================================================
# 1. Body
# ==========================================================================
def _vec3(x) -> np.ndarray:
    """Convert to a float array of shape (3,). 2-component input gets z = 0."""
    v = np.array(x, dtype=float).ravel()   # np.array copies, so we never alias the caller's data
    if v.size == 2:
        v = np.append(v, 0.0)              # allow 2-D setups: (x, y) -> (x, y, 0)
    if v.size != 3:
        raise ValueError(f"expected a 2- or 3-component vector, got {x!r}")
    return v


# eq=False: the auto-generated __eq__ would compare numpy arrays, which raises
# an "ambiguous truth value" error. We never compare bodies, so skip it.
@dataclass(eq=False)
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    # default_factory gives every Body its own zero array (a plain default would be shared)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = ""               # used for the plot legend
    color: Optional[str] = None  # any matplotlib color; None = automatic

    def __post_init__(self):
        # Normalize whatever the user passed (lists, tuples, 2-D...) into clean arrays.
        self.mass = float(self.mass)
        self.position = _vec3(self.position)
        self.velocity = _vec3(self.velocity)
        self.acceleration = _vec3(self.acceleration)

    def __repr__(self):
        label = self.name or "Body"
        return f"{label}(m={self.mass:g}, pos={self.position}, vel={self.velocity})"


def bodies_to_arrays(bodies: List[Body]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack bodies into arrays: positions (N,3), velocities (N,3), accelerations (N,3), masses (N,).

    Python objects are convenient for setting up a problem, but looping over them
    is slow. The physics runs on these packed arrays instead (vectorized NumPy).
    """
    pos = np.array([b.position for b in bodies])
    vel = np.array([b.velocity for b in bodies])
    acc = np.array([b.acceleration for b in bodies])
    mass = np.array([b.mass for b in bodies])
    return pos, vel, acc, mass


def write_back(bodies: List[Body], pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> None:
    """Copy array state back into the Body objects (the reverse of bodies_to_arrays)."""
    for i, b in enumerate(bodies):
        b.position = pos[i].copy()   # .copy() so each Body owns its data, not a view into pos
        b.velocity = vel[i].copy()
        b.acceleration = acc[i].copy()


# ==========================================================================
# 2. Physics
# ==========================================================================
def compute_accelerations(pos, mass, G=1.0, softening=0.0):
    """Direct all-pairs O(N^2) gravitational acceleration.

    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

    `softening` (eps) caps the force when two bodies get very close, which
    prevents huge, unstable accelerations in close encounters. Use 0 for exact
    point-mass gravity (fine for well-separated orbits like planets).
    """
    # Broadcasting trick to get every pairwise displacement at once:
    #   pos[None, :, :] has shape (1, N, 3) -> varies along j
    #   pos[:, None, :] has shape (N, 1, 3) -> varies along i
    # Their difference has shape (N, N, 3) with diff[i, j] = r_j - r_i,
    # i.e. the vector pointing from body i toward body j.
    diff = pos[None, :, :] - pos[:, None, :]

    # Squared distance for every pair: sum x^2 + y^2 + z^2 over the last axis.
    # Result is (N, N); softening^2 is added to every entry.
    dist2 = np.einsum("ijk,ijk->ij", diff, diff) + softening ** 2

    # The diagonal (i == j) has dist2 = 0, and 0 ** -1.5 would be infinity.
    # Put a harmless dummy value there, then zero those entries after the power
    # so a body never pulls on itself.
    np.fill_diagonal(dist2, 1.0)
    inv_d3 = dist2 ** -1.5                 # 1 / (distance^2 + eps^2)^(3/2)
    np.fill_diagonal(inv_d3, 0.0)          # no self-interaction

    # Weight each pair by the *source* mass m_j (mass[None, :] scales column j),
    # then sum the weighted displacement vectors over j for each body i:
    #   einsum("ij,ijk->ik") = for each i and each coordinate k, sum_j w[i,j] * diff[i,j,k]
    return G * np.einsum("ij,ijk->ik", inv_d3 * mass[None, :], diff)


def kinetic_energy(vel, mass):
    """KE = sum of 1/2 m v^2 over all bodies."""
    return 0.5 * np.sum(mass * np.einsum("ij,ij->i", vel, vel))   # einsum gives |v|^2 per body


def potential_energy(pos, mass, G=1.0, softening=0.0):
    """PE = -G * sum over pairs (i < j) of m_i m_j / distance. Uses the same softening as the forces."""
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + softening ** 2)
    # Each pair must be counted once, not twice: take only the upper triangle (i < j).
    i, j = np.triu_indices(len(mass), k=1)
    return -G * np.sum(mass[i] * mass[j] / dist[i, j])


def total_energy(pos, vel, mass, G=1.0, softening=0.0):
    """Kinetic + potential. For a correct simulation this should stay (nearly) constant,
    which makes it the best sanity check for your timestep and integrator."""
    return kinetic_energy(vel, mass) + potential_energy(pos, mass, G, softening)


def to_center_of_mass_frame(pos, vel, mass):
    """Return (pos, vel) shifted so the COM is at the origin with zero net momentum.

    Without this, a system with net momentum drifts across the screen forever.
    """
    total = mass.sum()
    com = (mass[:, None] * pos).sum(axis=0) / total        # mass-weighted mean position
    com_vel = (mass[:, None] * vel).sum(axis=0) / total    # mass-weighted mean velocity
    return pos - com, vel - com_vel


# ---- integrators ----------------------------------------------------------
# Each one advances (pos, vel) by one timestep dt. They trade accuracy against
# cost (number of force evaluations) and differ in how well they conserve energy.
def euler_step(pos, vel, acc, dt, accel_fn):
    """Explicit Euler, 1st order. Simple, but energy grows steadily. Good for comparison only."""
    new_pos = pos + dt * vel     # move using the *old* velocity
    new_vel = vel + dt * acc     # kick using the *old* acceleration
    return new_pos, new_vel, accel_fn(new_pos)


def leapfrog_step(pos, vel, acc, dt, accel_fn):
    """Kick-drift-kick leapfrog, 2nd order and symplectic: energy error stays bounded.

    One force evaluation per step. A good default for long orbital runs.
    """
    half_vel = vel + 0.5 * dt * acc            # kick: half a step of acceleration
    new_pos = pos + dt * half_vel              # drift: full step of motion with the half-step velocity
    new_acc = accel_fn(new_pos)                # forces at the new positions (the only evaluation)
    new_vel = half_vel + 0.5 * dt * new_acc    # kick: second half step with the new acceleration
    return new_pos, new_vel, new_acc


def rk4_step(pos, vel, acc, dt, accel_fn):
    """Classic 4th-order Runge-Kutta. Very accurate per step, but not symplectic,
    so energy drifts slowly over very long runs. Four force evaluations per step."""
    # We solve  dx/dt = v,  dv/dt = a(x)  by sampling the slope four times:
    # at the start, twice at the midpoint, and at the end. "kNx" is the position
    # slope (a velocity) and "kNv" is the velocity slope (an acceleration).
    h = 0.5 * dt
    k1x, k1v = vel, acc                        # slopes at the start (acc is reused, no new evaluation)
    k2x = vel + h * k1v                        # midpoint estimate using the k1 slopes
    k2v = accel_fn(pos + h * k1x)
    k3x = vel + h * k2v                        # second midpoint estimate, refined with k2
    k3v = accel_fn(pos + h * k2x)
    k4x = vel + dt * k3v                       # slopes at the end of the step, using k3
    k4v = accel_fn(pos + dt * k3x)
    # Weighted average of the four slopes (midpoint slopes count double).
    new_pos = pos + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    new_vel = vel + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return new_pos, new_vel, accel_fn(new_pos)   # 4th evaluation, cached for the next step


# Register new schemes here and they become available everywhere by name
# (including the --integrator option in main.py).
INTEGRATORS = {
    "euler": euler_step,
    "leapfrog": leapfrog_step,
    "rk4": rk4_step,
}


# ==========================================================================
# 3. Simulation
# ==========================================================================
@dataclass
class History:
    """A recorded run: used for energy plots or replaying/exporting trajectories."""
    times: np.ndarray       # (frames,)
    positions: np.ndarray   # (frames, N, 3)
    energies: np.ndarray    # (frames,)


class Simulation:
    def __init__(self, bodies: List[Body], G=1.0, softening=0.0, integrator="leapfrog"):
        if integrator not in INTEGRATORS:
            raise ValueError(f"unknown integrator {integrator!r}; choose from {sorted(INTEGRATORS)}")
        self.bodies = bodies
        self.G = G                       # gravitational constant in whatever units you chose
        self.softening = softening
        self.integrator_name = integrator
        self._integrator = INTEGRATORS[integrator]

        # State lives in packed arrays for speed; Body objects are updated by sync_bodies().
        # (The acceleration array from the bodies is ignored: we recompute it below.)
        self.pos, self.vel, _, self.mass = bodies_to_arrays(bodies)
        self.t = 0.0
        # Leapfrog and RK4 need the acceleration at the current positions before the first step.
        self.acc = self._accel(self.pos)

    @property
    def n(self) -> int:
        return len(self.mass)

    def _accel(self, pos):
        """Bind this simulation's masses, G and softening so integrators only need to pass positions."""
        return compute_accelerations(pos, self.mass, self.G, self.softening)

    def energy(self) -> float:
        """Total (kinetic + potential) energy of the current state."""
        return total_energy(self.pos, self.vel, self.mass, self.G, self.softening)

    def step(self, dt: float) -> None:
        """Advance the whole system by one timestep with the chosen integrator."""
        self.pos, self.vel, self.acc = self._integrator(self.pos, self.vel, self.acc, dt, self._accel)
        self.t += dt

    def run(self, steps: int, dt: float, record_every: int = 1) -> History:
        """Advance `steps` steps, recording state every `record_every` steps.

        Recording every step is wasteful for long runs; a larger `record_every`
        keeps memory small. The initial state is always recorded first.
        """
        times = [self.t]
        frames = [self.pos.copy()]       # .copy(): self.pos is replaced each step, but be explicit
        energies = [self.energy()]
        for s in range(1, steps + 1):
            self.step(dt)
            if s % record_every == 0:
                times.append(self.t)
                frames.append(self.pos.copy())
                energies.append(self.energy())   # an extra O(N^2) cost, only at recorded steps
        self.sync_bodies()
        return History(np.array(times), np.array(frames), np.array(energies))

    def center_of_mass_frame(self) -> None:
        """Shift to the frame where the COM is at rest at the origin."""
        self.pos, self.vel = to_center_of_mass_frame(self.pos, self.vel, self.mass)
        self.acc = self._accel(self.pos)   # positions moved, so refresh the cached accelerations

    def sync_bodies(self) -> None:
        """Copy the current array state back into the Body objects."""
        write_back(self.bodies, self.pos, self.vel, self.acc)
