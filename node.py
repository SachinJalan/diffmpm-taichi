from typing import Callable, Tuple
import taichi as ti
ti.init(arch=ti.gpu)

#is the initalized required for taichi as in jax it is used for jit compilation of jax
@ti.data_oriented
class Nodes:
    def __init__(
        self,
        nnodes: ti.i32,
        loc: ti.field,
        initialized: bool = None,
        data: Tuple[ti.field, ...] = tuple(),
    ):
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
        return self.nnodes
    
    def __repr__(self):
        return f"Node(nnodes={self.nnodes})"
    
    def get_total_force(self):
        return self.f_int + self.f_ext + self.f_damp
