import numpy as np
from utils.logger import logger

class PSOOptimizer:
    """
    Particle Swarm Optimizer to find the best parameters for a trading strategy.
    """
    def __init__(self, 
                 objective_function, 
                 param_bounds: dict, 
                 n_particles: int = 30, 
                 max_iter: int = 50,
                 w: float = 0.5, 
                 c1: float = 1.5, 
                 c2: float = 1.5):
        """
        Args:
            objective_function: A function that takes a dict of parameters and returns a score to maximize.
            param_bounds (dict): A dictionary with parameter names as keys and (min, max) tuples as values.
            n_particles (int): Number of particles in the swarm.
            max_iter (int): Maximum number of iterations.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        self.objective_function = objective_function
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_dims = len(self.param_names)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2

        # Initialize swarm
        self.particles_pos = np.random.rand(self.n_particles, self.n_dims)
        for i, key in enumerate(self.param_names):
            min_b, max_b = self.param_bounds[key]
            self.particles_pos[:, i] = self.particles_pos[:, i] * (max_b - min_b) + min_b

        self.particles_vel = np.random.rand(self.n_particles, self.n_dims) * 0.1
        self.pbest_pos = self.particles_pos.copy()
        self.pbest_score = np.full(self.n_particles, -np.inf)
        self.gbest_pos = None
        self.gbest_score = -np.inf

    def optimize(self):
        """Runs the PSO optimization loop."""
        logger.info(f"🚀 Starting PSO with {self.n_particles} particles for {self.max_iter} iterations...")

        for i in range(self.max_iter):
            for j in range(self.n_particles):
                params = {name: self.particles_pos[j, k] for k, name in enumerate(self.param_names)}
                score = self.objective_function(params)

                if score > self.pbest_score[j]:
                    self.pbest_score[j] = score
                    self.pbest_pos[j] = self.particles_pos[j].copy()

                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = self.particles_pos[j].copy()

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            self.particles_vel = (self.w * self.particles_vel +
                                  self.c1 * r1 * (self.pbest_pos - self.particles_pos) +
                                  self.c2 * r2 * (self.gbest_pos - self.particles_pos))
            self.particles_pos += self.particles_vel

            # Clamp positions to bounds
            for k, key in enumerate(self.param_names):
                min_b, max_b = self.param_bounds[key]
                self.particles_pos[:, k] = np.clip(self.particles_pos[:, k], min_b, max_b)

            logger.info(f"Iteration {i+1}/{self.max_iter} | Best Score (Sharpe): {self.gbest_score:.4f}")

        best_params = {name: self.gbest_pos[k] for k, name in enumerate(self.param_names)}
        return best_params, self.gbest_score