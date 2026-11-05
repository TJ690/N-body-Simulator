"""Plotting and animation with matplotlib.

This is the only file in the project that knows about matplotlib. To use a
different renderer (pygame, vpython, ...), write a replacement for `animate`
that reads `sim.pos`, calls `sim.step(dt)`, and draws; nothing else changes.

matplotlib is imported inside the functions so the rest of the project
(core.py) runs fine on a machine without it, e.g. headless.
"""

from collections import deque

import numpy as np


def _colors(bodies, plt):
    """One color per body: the body's own color if set, otherwise cycle through matplotlib's defaults."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [b.color or cycle[i % len(cycle)] for i, b in enumerate(bodies)]


def animate(sim, dt, frames, steps_per_frame=5, trail=200, limit=None, interval=20):
    """Step the simulation live and draw the x-y projection.

    frames           number of animation frames (total steps = frames * steps_per_frame)
    steps_per_frame  physics steps between redraws. Physics usually needs a much smaller
                     timestep than the eye needs frames, so several steps share one redraw.
    trail            how many frames of trail to draw per body (0 = none; skipped for N > 30)
    limit            half-width of the view box; default is 1.5x the initial extent
    interval         milliseconds between frames
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    n = sim.n
    colors = _colors(sim.bodies, plt)
    if limit is None:
        # Fixed view box so the picture does not jitter; max(..., 1e-12) avoids a zero-size box.
        limit = 1.5 * max(float(np.abs(sim.pos[:, :2]).max()), 1e-12)

    # Marker area grows with mass^(1/3) so a Sun does not hide everything else
    # while light bodies stay visible. Large clusters just get small uniform dots.
    if n <= 20:
        sizes = 15 + 85 * (sim.mass / sim.mass.max()) ** (1.0 / 3.0)
    else:
        sizes = 12

    # ---- static parts of the figure (created once) -----------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")    # equal axis scaling so circular orbits look circular
    ax.set_title(f"N-body (all-pairs), N = {n}, integrator = {sim.integrator_name}")

    # The artists below are the only things that change from frame to frame.
    dots = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], s=sizes, c=colors, zorder=3)
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")   # axes coords: top-left corner

    # Trails: one line per body, fed from a rolling buffer of recent positions.
    # With many bodies the lines just add clutter and slow drawing, so they are skipped.
    lines = []
    trail_buf = deque(maxlen=max(trail, 1))   # deque with maxlen drops the oldest entry automatically
    if trail > 0 and n <= 30:
        lines = [ax.plot([], [], lw=0.8, alpha=0.6, color=colors[i])[0] for i in range(n)]

    # Legend only if bodies are named and there are few enough to be readable.
    # Proxy Line2D handles are used because the single scatter call has no per-body labels.
    if any(b.name for b in sim.bodies) and n <= 12:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[i],
                   markeredgecolor=colors[i], label=b.name or f"body {i}")
            for i, b in enumerate(sim.bodies)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    # ---- per-frame update --------------------------------------------------
    def update(_frame):
        # Advance the physics, then redraw once.
        for _ in range(steps_per_frame):
            sim.step(dt)
        xy = sim.pos[:, :2]               # drop z: we draw the x-y projection
        dots.set_offsets(xy)
        if lines:
            trail_buf.append(xy.copy())   # copy: sim.pos is replaced next step, but be explicit
            arr = np.asarray(trail_buf)   # shape (T, N, 2): T recent frames
            for i, line in enumerate(lines):
                line.set_data(arr[:, i, 0], arr[:, i, 1])
        time_text.set_text(f"t = {sim.t:.3f}")
        # With blit=True matplotlib redraws only the artists returned here, which is much faster.
        return [dots, time_text, *lines]

    # Keep the FuncAnimation object alive (we return it): if it is garbage-collected, the animation stops.
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, repeat=False)
    plt.show()
    return anim


def plot_energy(histories):
    """Plot |relative energy drift| vs time for one or more runs.

    histories: dict mapping a label (e.g. integrator name) to a core.History
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, h in histories.items():
        # |E(t) - E(0)| / |E(0)|: zero means perfect conservation.
        drift = np.abs((h.energies - h.energies[0]) / h.energies[0])
        # Log scale shows the huge range between integrators; the floor avoids log(0) for exact runs.
        ax.semilogy(h.times, np.maximum(drift, 1e-16), label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("|ΔE / E₀|")
    ax.set_title("Energy conservation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()
