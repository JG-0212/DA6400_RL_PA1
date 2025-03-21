import numpy as np

# Based on the description in section 9.5.4 RLBook


class TileCoder:

    def __init__(self, num_tiles_per_dim,  num_tilings, lower_lim, upper_lim):
        """
        To tile-code continuous states into discrete states

        Args:
            - num_tiles_per_dim (ArrayLike): Number of divisions for discretizing each dimension.
            - num_tilings (int): Total number of overlapping tiling grids.
            - lower_lim (ArrayLike): Lower bounds of the state space.
            - upper_lim (ArrayLike): Upper bounds of the state space.
        """

        self.num_tiles_per_dim = np.array(num_tiles_per_dim, dtype=np.int32)
        self.num_tilings = num_tilings

        self.lower_lim = np.array(lower_lim)
        self.upper_lim = np.array(upper_lim)
        self.range = self.upper_lim - self.lower_lim

        self.feature_dim = self.num_tiles_per_dim.shape[0]

        self.tile_width = self.range/self.num_tiles_per_dim

        # Computing vectors to offset the tilings
        # Miller and Glanz (1996) recommend using displacement vectors consisting of the first odd integers.
        base_offset = 2*np.arange(self.feature_dim) + 1
        self.offsets = np.array([(k
                                 * base_offset
                                 * (self.tile_width/self.num_tilings))
                                 for k in range(num_tilings)])

        # Index based hashing
        tmp = np.append(self.num_tiles_per_dim[1:], 1)
        self.base_hash = np.cumprod(tmp[::-1])[::-1]

    def __call__(self, x):
        """
        Args:
            - x : Continuous state to be discretized.

        Returns:
            - tuple : A tuple representing the discrete state.
        """
        eps = 1e-10
        tile_coords = ((x - self.lower_lim) / (self.tile_width + eps)
                       + self.offsets).astype(int)

        tile_coord_hash = np.dot(self.base_hash, tile_coords.T)
        return tuple(tile_coord_hash)


if __name__ == '__main__':

    lower_lim = np.array([0, 0, 0])
    upper_lim = np.array([4, 3, 2])
    num_tilings = 1

    tc = TileCoder(
        num_tiles_per_dim=[4, 3, 2],
        num_tilings=num_tilings,
        lower_lim=lower_lim,
        upper_lim=upper_lim
    )

    print(tc([0, 0, 0]))
