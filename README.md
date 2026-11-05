N-Body Simulator

A small gravitational N-body simulator written in Python. It uses the direct all-pairs method: every body's acceleration is computed from every other body, so each force evaluation costs O(N²). That is simple and exact, and works well for small systems (up to a few thousand bodies).

Features
Vectorized NumPy force calculation with optional Plummer softening
Three selectable time integrators: Euler, Leapfrog (symplectic), and RK4
Built-in presets: figure-eight three-body orbit, a Sun + five planets solar system, and a random virialized star cluster
Live matplotlib animation with orbit trails and a legend
Energy-conservation diagnostics and a side-by-side integrator comparison plot
Requirements
Python 3.9 or newer
NumPy and matplotlib
Setup
bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
Usage
bash
python main.py                               # figure-eight orbit, animated
python main.py --preset solar                # Sun + Mercury through Jupiter
python main.py --preset cluster --n 200      # random cluster with 200 bodies
python main.py --integrator rk4              # euler | leapfrog | rk4
python main.py --preset solar --no-plot      # headless run, prints energy drift
python main.py --compare                     # plot energy drift for all integrators
Options
Option	Description	Default
--preset	figure-eight, solar, or cluster	figure-eight
--integrator	euler, leapfrog, or rk4	leapfrog
--n	Number of bodies (cluster preset only)	100
--seed	Random seed (cluster preset only)	0
--dt	Timestep	preset value
--steps	Total number of steps	preset value
--no-plot	Run without a window and print the energy drift	off
--compare	Run all integrators and plot their energy drift	off
Project structure
core.py          Body class, gravity and energy functions, integrators, Simulation class
visualizer.py    matplotlib animation and energy plots (the only file that imports matplotlib)
main.py          Presets, command-line interface, and the entry point
requirements.txt Python dependencies
How it works

For each body i, the acceleration is

a_i = G * sum over j != i of  m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

where eps is the softening length (set it to 0 for exact point-mass gravity). The sum is evaluated for all pairs at once using NumPy broadcasting.

Integrators
Integrator	Order	Force evaluations per step	Notes
Euler	1st	1	Energy grows steadily; included for comparison
Leapfrog	2nd	1	Symplectic, so energy error stays bounded; best default for orbits
RK4	4th	4	Very accurate per step; energy drifts slowly over very long runs

Relative energy drift on the figure-eight orbit over 6.5 time units (dt = 0.001):

Integrator	Drift
Euler	3 × 10⁻²
Leapfrog	1 × 10⁻⁷
RK4	3 × 10⁻¹⁴
Extending
New integrator: write a function with the same signature as leapfrog_step in core.py and add it to the INTEGRATORS dict. It then appears in --integrator automatically.
New preset: add a function in main.py that returns (bodies, config) and register it in PRESETS.
Different renderer: replace animate in visualizer.py. Nothing else depends on matplotlib.
