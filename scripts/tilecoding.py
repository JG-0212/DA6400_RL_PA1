import numpy as np

# Based on the description in section 9.5.4 RLBook


class TileCoder:

    def __init__(self, num_tiles_per_feature,  num_tilings, lower_lim, upper_lim):
        """
        To tile-code continuous states into discrete states

        Args:
            - num_tiles_per_feature (ArrayLike): Number of divisions for discretizing each dimension.
            - num_tilings (int): Total number of overlapping tiling grids.
            - lower_lim (ArrayLike): Lower bounds of the state space.
            - upper_lim (ArrayLike): Upper bounds of the state space.
        """

        self.num_tiles_per_feature = np.array(num_tiles_per_feature, 
                                              dtype=np.int32)
        self.num_tilings = num_tilings

        self.lower_lim = np.array(lower_lim)
        self.upper_lim = np.array(upper_lim)
        self.range = self.upper_lim - self.lower_lim

        self.feature_dim = self.num_tiles_per_feature.shape[0]

        self.tile_width = self.range/self.num_tiles_per_feature
        self.base_tile_index = np.prod(
            self.num_tiles_per_feature)*np.arange(self.num_tilings)

        # Computing vectors to offset the tilings
        # Miller and Glanz (1996) recommend using displacement vectors consisting of the first odd integers.
        base_offset = 2*np.arange(self.feature_dim) + 1
        self.offsets = np.array([(k
                                 * base_offset
                                 * (self.tile_width/self.num_tilings))
                                 for k in range(num_tilings)])

        # Index based hashing
        tmp = np.append(self.num_tiles_per_feature[1:], 1)
        self.base_hash = np.cumprod(tmp[::-1])[::-1]

    def __call__(self, x):
        """
        Args:
            - x : Continuous state to be discretized.

        Returns:
            - list : A list representing the discrete state.
        """
        eps = 1e-10
        tile_coords = ((x - self.lower_lim) / (self.tile_width + eps)
                       + self.offsets).astype(np.int32)


        tile_coord_hash = (
            self.tile_base_index
            + np.dot(self.base_hash, tile_coords.T)
        )
        return list(self.base_tile_index + tile_coord_hash)

    @property
    def total_tiles(self):
        return int(self.num_tilings*np.prod(self.num_tiles_per_feature))
    

class QTable:

    def __init__(self, num_tiles_per_feature,  num_tilings, lower_lim, upper_lim, action_size):

        self.tile_coder = TileCoder(
            num_tiles_per_feature=num_tiles_per_feature,
            num_tilings=num_tilings,
            lower_lim=lower_lim,
            upper_lim=upper_lim
        )

        self.table_size = np.append(self.tile_coder.total_tiles,
                                    action_size)

        self.table = np.zeros(self.table_size)

    def __getitem__(self, key):
        """To get q_value = q_table[state] or q_value = q_table[state, action]
        """
        if isinstance(key, tuple) and len(key) == 2:
            state, action = key
            idx_s = self.tile_coder(state)
            return self.table[idx_s, action].mean()
        else:
            state = key
            idx_s = self.tile_coder(state)
            return self.table[idx_s].mean(axis=0)

    def __setitem__(self, key, value):
        """To set q_table[state, action] = value
        """
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError("Key must be a tuple (state, action)")

        state, action = key
        idx_s = self.tile_coder(state)
        self.table[idx_s, action] = value


if __name__ == '__main__':

    lower_lim = np.array([0, 0, 0])
    upper_lim = np.array([8, 4, 6])
    num_tilings = 1

    tc = TileCoder(
        num_tiles_per_feature=[4, 2, 3],
        num_tilings=num_tilings,
        lower_lim=lower_lim,
        upper_lim=upper_lim
    )

    for i in range(4):
        for j in range(2):
            for k in range(3):
                print(tc([2*i+0.01, 2*j+0.01, 2*k+0.01])[0])