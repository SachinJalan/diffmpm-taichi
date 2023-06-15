from typing import Callable, Tuple
import taichi as ti

ti.init(arch=ti.gpu)


# is the initalized required for taichi as in jax it is used for jit compilation of jax
@ti.data_oriented
class Nodes:
    """
    Nodes container class.

    Keeps track of all values required for nodal points.

    Attributes
    ----------
    nnodes : int
        Number of nodes stored.
    loc : array_like
        Location of all the nodes.
    velocity : array_like
        Velocity of all the nodes.
    mass : array_like
        Mass of all the nodes.
    momentum : array_like
        Momentum of all the nodes.
    f_int : array_like
        Internal forces on all the nodes.
    f_ext : array_like
        External forces present on all the nodes.
    f_damp : array_like
        Damping forces on the nodes.
    """

    def __init__(
        self,
        nnodes: ti.i32,
        loc: ti.field,
        initialized: bool = None,
        data: Tuple[ti.field, ...] = tuple(),
    ):
        """
        Initialize container for Nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes stored.
        loc : array_like
            Locations of all the nodes. Expected shape (nnodes, 1, ndim)
        initialized: bool
            False if node property arrays like mass need to be initialized.
        If True, they are set to values from `data`.
        data: tuple
            Tuple of length 7 that sets arrays for mass, density, volume,
        """

        self.nnodes = nnodes
        if len(loc.shape) != 3:
            raise ValueError(
                f"`loc` should be of size (nnodes, 1, ndim); found {loc.shape}"
            )
        self.loc = ti.field(ti.f32, loc.shape)
        if initialized is None:
            self.velocity = ti.field(ti.f32, loc.shape)
            self.velocity.fill(0)
            self.acceleration = ti.field(ti.f32, loc.shape)
            self.acceleration.fill(0)
            self.mass = ti.field(ti.f32, loc.shape)
            self.mass.fill(0)
            self.momentum = ti.field(ti.f32, loc.shape)
            self.momentum.fill(0)
            self.f_int = ti.field(ti.f32, loc.shape)
            self.f_int.fill(0)
            self.f_ext = ti.field(ti.f32, loc.shape)
            self.f_ext.fill(0)
            self.f_damp = ti.field(ti.f32, loc.shape)
            self.f_damp.fill(0)
        else:
            (
                self.velocity,
                self.acceleration,
                self.mass,
                self.momentum,
                self.f_int,
                self.f_ext,
                self.f_damp,
            ) = data
        self.initialized = True

    def reset_values(self):
        """Reset nodal parameter values except location."""
        self.velocity.fill(0)
        self.acceleration.fill(0)
        self.mass.fill(0)
        self.momentum.fill(0)
        self.f_int.fill(0)
        self.f_ext.fill(0)
        self.f_damp.fill(0)

    def __len__(self):
        """Set length of class as number of nodes."""
        return self.nnodes

    def __repr__(self):
        """Repr containing number of nodes."""
        return f"Node(nnodes={self.nnodes})"

    def get_total_force(self):
        """Calculate total force on the nodes."""
        tot_f = ti.field(ti.f32, self.f_int.shape)

        @ti.kernel
        def fill_tot_f():
            for i in range(self.f_int.shape[0]):
                tot_f[i] = self.f_int[i] + self.f_ext[i] + self.f_damp[i]

        fill_tot_f()
        return tot_f
