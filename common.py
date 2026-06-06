#file will contain the common functions and classes in all the steps
import numpy as np
import matplotlib.pyplot as plt


#x: Positions of the particles in 3D space
#y: Velocities of the particles in 3D space
#m: Masses of the particles
#G gravitational Constant
class System:
    def __init__(
            self, num_particles: int, x: np.ndarray, v:  np.ndarray, m: np.ndarray, G: float
    ) -> None:
        self.num_particles = num_particles
        self.x = x
        self.v = v
        self.m = m
        self.G = G
