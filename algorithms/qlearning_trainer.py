# algorithms/qlearning_trainer.py
import numpy as np
import math
# ============================================================
# Discretização da Observação
# ============================================================
class FarolObservationDiscretizer:
    def __init__(self):
        # reduzir número de bins para manter o espaço de estados manejável
        self.ray_bins = [3]*8
        self.radar_bins = [2]*4
        self.bins = self.ray_bins + self.radar_bins
        self.low = [0.0]*8 + [0.0]*4
        self.high = [5.0]*8 + [1.0]*4

    def discretize(self, obs):
        indices = []
        for i in range(len(obs)):
            val = np.clip(obs[i], self.low[i], self.high[i])
            n = self.bins[i]
            bin_size = (self.high[i] - self.low[i]) / n
            idx = int((val - self.low[i]) / bin_size)
            idx = min(idx, n-1)
            indices.append(idx)
        return tuple(indices)

    def tuple_to_index(self, tup):
        idx = 0
        for i, val in enumerate(tup):
            mult = np.prod(self.bins[i+1:]) if i+1 < len(self.bins) else 1
            idx += val * mult
        return int(idx)

    @property
    def n_states(self):
        return int(np.prod(self.bins))
    
class MazeObservationDiscretizer:
    def __init__(self, bins_pos=10, bins_dist=10):
        self.bins_pos = bins_pos
        self.bins_dist = bins_dist
        self.n_states = bins_pos * bins_pos * bins_dist

    def discretize(self, obs):
        # obs já é um vetor NumPy
        pos_x_norm = obs[8]  # índice 8 → pos_x_norm
        pos_y_norm = obs[9]  # índice 9 → pos_y_norm
        dist_norm  = obs[10] # índice 10 → dist_norm

        # discretizar
        pos_x_bin = min(int(pos_x_norm * self.bins_pos), self.bins_pos - 1)
        pos_y_bin = min(int(pos_y_norm * self.bins_pos), self.bins_pos - 1)
        dist_bin  = min(int(dist_norm * self.bins_dist), self.bins_dist - 1)

        return (pos_x_bin, pos_y_bin, dist_bin)

    def tuple_to_index(self, discretized_tuple):
        pos_x_bin, pos_y_bin, dist_bin = discretized_tuple
        index = pos_x_bin
        index += pos_y_bin * self.bins_pos
        index += dist_bin * self.bins_pos * self.bins_pos
        return index