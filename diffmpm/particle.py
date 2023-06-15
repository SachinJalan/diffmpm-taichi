from typing import Tuple
import taichi as ti

from diffmpm.element import _Element
from diffmpm.material import Material

ti.init(arch=ti.gpu)


class Particles:
    def __init__(
        self,
        loc: ti.field,
        material: Material,
        element_ids=ti.field,
        data: Tuple[ti.field, ...] = None,
    ):
        self.material = material
        self.element_ids = element_ids
        if len(loc.shape) != 3:
            raise ValueError(
                f"`loc` should be of size (nparticles, 1, ndim); " f"found {loc.shape}"
            )
        self.loc = loc
        self.mass = ti.field(ti.f32, shape=(self.loc.shape[0], 1, 1))
        self.mass.fill(1)
        self.density = ti.field(ti.f32, shape=(self.loc.shape[0], 1, 1))
        self.density.fill(self.material.properties["density"])
        self.volume = ti.field(ti.f32, shape=(self.loc.shape[0], 1, 1))
        self.volume.fill(1 / (self.material.properties["density"]))
        self.velocity = ti.field(ti.f32, shape=(self.loc.shape[0], 1, 1))
        self.velocity.fill(0)
        self.acceleration = ti.field(ti.f32, shape=(self.loc.shape[0], 1, 1))
        self.acceleration.fill(0)
        self.momentum = ti.field(ti.f32, shape=(self.loc.shape[0], 6, 1))
        self.momentum.fill(0)
        self.strain = ti.field(ti.f32, shape=(self.loc.shape[0], 6, 1))
        self.strain.fill(0)
        self.stress = ti.field(ti.f32, shape=(self.loc.shape[0], 6, 1))
        self.stress.fill(0)
        self.strain_rate = ti.field(ti.f32, shape=(self.loc.shape[0], 6, 1))
        self.strain_rate.fill(0)
        self.dstrain = ti.field(ti.f32, shape=(self.loc.shape[0], 6, 1))
        self.dstrain.fill(0)
        self.f_ext = ti.field(ti.f32, shape=self.loc)
        self.f_ext.fill(0)
        self.reference_loc = ti.field(ti.f32, shape=self.loc)
        self.reference_loc.fill(0)
        self.volume_strain_centroid = ti.field(ti.f32, shape=(self.loc.shape[0], 1))
        self.volume_strain_centroid.fill(0)

    def __len__(self):
        return self.loc.shape[0]

    def __repr__(self):
        return f"Particles(nparticles={len(self)})"

    def set_mass_volume(self, m: ti.f32 | ti.field):
        if m.shape == self.mass.shape:
            self.mass = m
        elif isinstance(m, ti.f32):
            self.mass.fill(m)
        else:
            raise ValueError(
                f"Incompatible shapes. Expected {self.mass.shape}, " f"found {m.shape}."
            )

        @ti.kernel
        def _set_volume():
            for i in self.loc.shape[0]:
                self.volume[i, 1, 1] = (
                    self.mass[i, 1, 1] / self.material.properties["density"]
                )

        _set_volume()

    def update_natural_coords(self, elements: _Element):
        t = elements.id_to_node_loc(self.element_ids)
        xi_coords = ti.field(ti.f32, shape=self.loc.shape)

        @ti.kernel
        def fill_xi_coords():
            for i in range(self.loc.shape[0]):
                for j in range(self.loc.shape[2]):
                    xi_coords[i, 0, j] = self.loc[i, 0, j] - (
                        (t[i, 0, j] + t[i, 1, j]) / 2
                    ) * (2 / (t[i, 1, j] - t[i, 0, j]))

        fill_xi_coords()
        self.reference_loc = xi_coords

    def update_position_velocity(self, elements: _Element, dt: ti.f32):
        mapped_positions = elements.shapefn(self.reference_loc)
        mapped_ids = ti.field(ti.i32, shape=(self.element_ids.shape[0], 2))

        @ti.kernel
        def fill_mapped_ids():
            for i in range(self.element_ids.shape[0]):
                tmp_ids = elements.id_to_node_ids(self.element_ids[i])[0]
                mapped_ids[i, 0] = tmp_ids[0]
                mapped_ids[i, 1] = tmp_ids[1]

        fill_mapped_ids()
        total_force = elements.nodes.get_total_force()

        @ti.kernel
        def update_velocity():
            for i in range(self.velocity.shape[0]):
                for j in range(self.velocity.shape[2]):