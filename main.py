"""Entry point: choose initial conditions, wire the modules together, run.

Examples
    python main.py                               # figure-eight 3-body orbit, animated
    python main.py --preset solar                # Sun + 5 planets (units: AU, years, solar masses)
    python main.py --preset cluster --n 200      # random virialized cluster
    python main.py --integrator rk4              # euler | leapfrog | rk4
    python main.py --preset solar --no-plot      # headless run, prints energy drift
    python main.py --compare                     # plot energy drift of all integrators

To add your own setup: write a function like `figure_eight()` that returns
(bodies, config) and register it in PRESETS below.
"""

import argparse

import numpy as np

import visualizer
from core import INTEGRATORS, Body, Simulation, potential_energy


# --------------------------------------------------------------------------
# Presets. Each returns (bodies, config) where config holds the physical
# constants and sensible run/plot settings for that setup:
#   G, softening     physics parameters passed to Simulation
#   dt, steps        timestep and number of steps for a full run
#   steps_per_frame  physics steps per animation frame
#   trail, limit     trail length (frames) and half-width of the view box
# --------------------------------------------------------------------------
def figure_eight():
    """Chenciner-Montgomery figure-eight orbit: three equal masses chasing each other
    around a single figure-8 path. A famous exact solution, so it makes a great test:
    a good integrator brings the bodies back to where they started after one period (~6.33).
    """
    # Published initial conditions (G = 1, unit masses): bodies A and B start
    # symmetric about the origin, body C starts at the origin moving fast,
    # and A and B each move at -1/2 of C's velocity so total momentum is zero.
    p1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    bodies = [
        Body(1.0, p1, -v3 / 2, name="A"),
        Body(1.0, -p1, -v3 / 2, name="B"),
        Body(1.0, np.zeros(3), v3, name="C"),
    ]
    # softening must be 0 here: the exact solution assumes unsoftened gravity.
    cfg = dict(G=1.0, softening=0.0, dt=0.001, steps=6500,
               steps_per_frame=10, trail=300, limit=1.5)
    return bodies, cfg


def solar_system():
    """Sun + Mercury..Jupiter on circular coplanar orbits.

    Units: AU, years, solar masses, so G = 4 pi^2. (Earth orbits at 1 AU in 1 year,
    which fixes G * M_sun = 4 pi^2.) Real orbits are slightly elliptical; circular is
    a simple, stable approximation that still shows the planets' relative motion.
    """
    G = 4.0 * np.pi ** 2
    # name, mass [Msun], orbital radius [AU], start angle [rad], color
    # Angles are spread out so the planets do not all start in a line.
    table = [
        ("Sun",     1.0,      0.0,   0.0, "gold"),
        ("Mercury", 1.660e-7, 0.387, 0.0, "gray"),
        ("Venus",   2.447e-6, 0.723, 0.5, "tab:orange"),
        ("Earth",   3.003e-6, 1.000, 2.0, "tab:blue"),
        ("Mars",    3.227e-7, 1.524, 3.5, "tab:red"),
        ("Jupiter", 9.546e-4, 5.203, 5.0, "tab:brown"),
    ]
    bodies = []
    for name, m, r, theta, color in table:
        if r == 0.0:
            pos, vel = np.zeros(3), np.zeros(3)   # the Sun starts at rest at the origin
        else:
            # Circular orbit speed around a 1-solar-mass star: v = sqrt(G M / r).
            v_circ = np.sqrt(G * 1.0 / r)
            pos = r * np.array([np.cos(theta), np.sin(theta), 0.0])
            # Velocity is perpendicular to the radius vector (tangential direction).
            vel = v_circ * np.array([-np.sin(theta), np.cos(theta), 0.0])
        bodies.append(Body(m, pos, vel, name=name, color=color))
    # dt = 0.001 yr gives Mercury (period 0.24 yr) ~240 steps per orbit.
    cfg = dict(G=G, softening=0.0, dt=0.001, steps=24000,
               steps_per_frame=20, trail=150, limit=6.0)
    return bodies, cfg


def random_cluster(n=100, seed=0):
    """Gaussian blob of equal masses (total mass 1, G = 1) in virial equilibrium."""
    rng = np.random.default_rng(seed)   # seeded so runs are reproducible
    softening = 0.05                    # needed here: random bodies can pass very close to each other
    pos = rng.normal(size=(n, 3))
    mass = np.full(n, 1.0 / n)
    vel = rng.normal(size=(n, 3))
    # Virial equilibrium: a bound, self-gravitating system is stable (neither
    # collapsing nor flying apart) when 2*KE = -PE. Scale the random velocities
    # by the factor that makes this true.
    ke = 0.5 * np.sum(mass * np.einsum("ij,ij->i", vel, vel))
    pe = potential_energy(pos, mass, 1.0, softening)
    vel *= np.sqrt(-0.5 * pe / ke)
    bodies = [Body(mass[i], pos[i], vel[i]) for i in range(n)]
    cfg = dict(G=1.0, softening=softening, dt=0.01, steps=3000,
               steps_per_frame=3, trail=0, limit=4.0)
    return bodies, cfg


# Name used on the command line -> function that builds the preset.
PRESETS = {
    "figure-eight": figure_eight,
    "solar": solar_system,
    "cluster": random_cluster,
}


# --------------------------------------------------------------------------
def build_simulation(preset, integrator, n=100, seed=0):
    """Create a ready-to-run Simulation (and its config) from a preset name."""
    if preset == "cluster":
        bodies, cfg = random_cluster(n, seed)   # only this preset takes parameters
    else:
        bodies, cfg = PRESETS[preset]()
    sim = Simulation(bodies, G=cfg["G"], softening=cfg["softening"], integrator=integrator)
    # Remove any net drift so the system stays centered in the plot. This also gives the
    # Sun its tiny recoil velocity in the solar-system preset.
    sim.center_of_mass_frame()
    return sim, cfg


def report(sim, e0):
    """Print how well energy was conserved. e0 is the energy before the run."""
    e1 = sim.energy()
    print(f"preset N = {sim.n}, integrator = {sim.integrator_name}, t = {sim.t:.3f}")
    print(f"energy: start {e0:.8f}, end {e1:.8f}, relative drift {abs((e1 - e0) / e0):.2e}")


def parse_args():
    ap = argparse.ArgumentParser(description="Direct all-pairs N-body simulator")
    ap.add_argument("--preset", choices=list(PRESETS), default="figure-eight")
    ap.add_argument("--integrator", choices=sorted(INTEGRATORS), default="leapfrog")
    ap.add_argument("--n", type=int, default=100, help="body count (cluster preset only)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (cluster preset only)")
    ap.add_argument("--dt", type=float, default=None, help="timestep (default: preset value)")
    ap.add_argument("--steps", type=int, default=None, help="total steps (default: preset value)")
    ap.add_argument("--no-plot", action="store_true", help="run headless and print energy drift")
    ap.add_argument("--compare", action="store_true",
                    help="run every integrator headless and plot their energy drift")
    return ap.parse_args()


def main():
    args = parse_args()
    sim, cfg = build_simulation(args.preset, args.integrator, args.n, args.seed)
    # Command-line values override the preset's defaults.
    dt = args.dt or cfg["dt"]
    steps = args.steps or cfg["steps"]
    # Record roughly 500 points per run for the energy plot, however long the run is.
    record_every = max(1, steps // 500)

    if args.compare:
        # Run the same problem with every integrator. A fresh simulation is built each
        # time because run() leaves the Body objects in their final state.
        histories = {}
        for name in sorted(INTEGRATORS):
            s, _ = build_simulation(args.preset, name, args.n, args.seed)
            histories[name] = s.run(steps, dt, record_every)
            e = histories[name].energies
            print(f"{name:9s} final relative drift {abs((e[-1] - e[0]) / e[0]):.2e}")
        visualizer.plot_energy(histories)
        return

    e0 = sim.energy()
    if args.no_plot:
        sim.run(steps, dt, record_every)
    else:
        # The animation steps the simulation live, so there is no separate run() here.
        spf = cfg["steps_per_frame"]
        visualizer.animate(sim, dt, frames=steps // spf, steps_per_frame=spf,
                           trail=cfg["trail"], limit=cfg["limit"])
    report(sim, e0)


if __name__ == "__main__":
    main()
