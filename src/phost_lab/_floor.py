"""A kernel that does nothing, to measure what replaying a graph costs by itself. Imports Warp, so it is loaded lazily."""

import warp as wp


@wp.kernel
def touch(x: wp.array(dtype=float)):
    x[0] = x[0] + 0.0
