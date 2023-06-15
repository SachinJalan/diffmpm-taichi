import abc
import itertools
from typing import Sequence, Tuple

import taichi as ti
from diffmpm.node import Nodes
import numpy as np

ti.init(arch=ti.gpu)


class _Element(abc.ABC):
    @abc.abstractmethod
    def id_to_node_ids(self):
        ...

    @abc.abstractmethod
    def id_to_node_loc(self):
        ...

    @abc.abstractmethod
    def id_to_node_vel(self):
        ...

    @abc.abstractmethod
    def shapefn(self):
        ...

    @abc.abstractmethod
    def shapefn_grad(self):
        ...


@ti.data_oriented
class Linear1D(_Element):
    """
    Container for 1D line elements (and nodes).

    Element ID:            0     1     2     3
    Mesh:               +-----+-----+-----+-----+
    Node IDs:           0     1     2     3     4

    + : Nodes
    +-----+ : An element
    """

    def __init__(
        self,
        nelements: ti.i32,
        el_len: ti.f32,
        boundary_nodes: Sequence,
        nodes: Nodes = None,
    ):
        """Initialize Linear1D.

        Arguments
        ---------
        nelements : int
            Number of elements.
        el_len : float
            Length of each element.
        boundary_nodes : Sequence
            IDs of nodes that are supposed to be fixed (boundary).
        """
        self.nelements = nelements
        self.ids = ti.field(ti.i32, (nelements))
        self.ids.from_numpy(np.arange(nelements))
        self.el_len = el_len
        if nodes is None:
            f = ti.field(ti.f32, shape=(nelements + 1, 1, 1))
            f.from_numpy((np.arange(nelements + 1).reshape(-1, 1, 1)) * el_len)
            self.nodes = Nodes(nelements + 1, f)
        else:
            self.nodes = nodes
        self.boundary_nodes = boundary_nodes

    def id_to_node_loc(self, id: ti.i32) -> ti.field:
        """
        Node locations corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        taichi.field
            Nodal locations for the element. Shape of returned
        array is (len(id), 2, 1)
        """
        temp_nparr = (
            np.array([self.nodes.loc[id], self.nodes.loc[id + 1]])
            .transpose(1, 0, 2, 3)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=temp_nparr.shape)
        f.from_numpy(temp_nparr)
        return f

    def id_node_ids(self, id: ti.i32) -> ti.field:
        """
        Node IDs corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        taichi.field
            Nodal IDs of the element. Shape of returned
        array is (2, 1)
        """
        temp_nparr = (
            np.array([self.nodes.loc[id], self.nodes.loc[id + 1]])
            .reshape(2, 1)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=(2, 1))
        f.from_numpy(temp_nparr)
        return f

    def id_to_node_vel(self, id: ti.i32) -> ti.field:
        """
        Node velocities corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        taichi.field
            Nodal velocities for the element. Shape of returned
        array is (2, 1)
        """
        temp_nparr = (
            np.array([self.nodes.velocity[id], self.nodes.velocity[id + 1]])
            .reshape(2, 1)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=(2, 1))
        f.from_numpy(temp_nparr)
        return f

    def shapefn(self, xi: float | ti.field):
        """
        Evaluate linear shape function.

        Arguments
        ---------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
        the function at. Expected shape is (npoints, 1, ndim)

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
        array will depend on the input shape. For example, in the linear
        case, if the input is a scalar, the returned array will be of
        the shape (1, 2, 1) but if the input is a vector then the output will
        be of the shape (len(x), 2, 1).
        """
        if len(xi.shape) != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        tmp_xi = xi.to_numpy()
        tmp_nparray = np.array([0.5 * (1 - tmp_xi), 0.5 * (1 + tmp_xi)]).transpose(
            1, 0, 2, 3
        )
        result = ti.field(ti.i32, shape=tmp_nparray.shape)
        result.from_numpy(tmp_nparray)
        return result

    def _shapefn_natural_grad(self, xi: float | np.ndarray):
        """
        Calculate the gradient of shape function.

        This calculation is done in the natural coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations of particles in natural coordinates to evaluate
        the function at.

        Returns
        -------
        array_like
            Evaluated gradient values of the shape function. The shape of
        the returned array will depend on the input shape. For example,
        in the linear case, if the input is a scalar, the returned array
        will be of the shape (1, 2) but if the input is a vector then the
        output will be of the shape (len(x), 2).
        """
        # TODO:Implement using automatic differentiation currently it is manual
        if len(xi.shape) != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = ti.field(ti.i32, shape=(xi.shape[0], 2))
        for i in range(xi.shape[0]):
            result[i, 0] = -0.5
            result[i, 1] = 0.5
        return result

    def shapefn_grad(self, xi: float | np.ndarray, coords: np.ndarray):
        """
        Gradient of shape function in physical coordinates.

        Arguments
        ---------
        xi : float, array_like
            Locations of particles to evaluate in natural coordinates.
        Expected shape (npoints, 1, ndim).
        coords : array_like
            Nodal coordinates to transform by. Expected shape
        (npoints, 1, ndim)

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `xi`
        """
        # TODO: See if it can be done using taichi fields
        if len(xi.shape) != 3:
            raise ValueError(
                f"`x` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        grad_sf = self._shapefn_natural_grad(xi)
        grad_sf_np = grad_sf.to_numpy()
        _jacobian = grad_sf_np @ coords
        result = grad_sf_np @ np.linalg.inv(_jacobian)
        result_taichi = ti.field(ti.f32, shape=result.T.shape)
        result_taichi.from_numpy(result.T)
        return result_taichi

    def set_particle_element_ids(self, particles):
        dimen = self.nodes.loc.shape[2]
        """
        Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @ti.kernel
        def fill_ids():
            for i in range(len(particles.loc)):
                particles.element_ids[i] = -1
                for j in range(len(self.nodes.loc) - 1):
                    if (
                        self.nodes.loc[j, 0, dimen - 1]
                        <= particles.loc[i, 0, dimen - 1]
                        and self.nodes.loc[j + 1, 0, dimen - 1]
                        > particles.loc[i, 0, dimen - 1]
                    ):
                        particles.element_ids[i] = self.nodes.loc[j, 0, dimen - 1]

        fill_ids()
    # Mapping from particles to nodes (P2G)
    def compute_nodal_mass(self, particles):
        r"""
        Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`(m)_i = \sum_p N_i(x_p) m_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()

        @ti.kernel
        def fill_mass():
            for i in range(len(particles)):
                self.nodes.mass[mapped_nodes[i, 0]] += (
                    particles.mass[i] * mapped_positions[i, 0, 0]
                )
                self.nodes.mass[mapped_nodes[i, 1]] += (
                    particles.mass[i] * mapped_positions[i, 1, 0]
                )

        fill_mass()

    def compute_nodal_momentum(self, particles):
        r"""
        Compute the nodal momentum based on particle momentum.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`(mv)_i = \sum_p N_i(x_p) (mv)_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()

        @ti.kernel
        def fill_momentum():
            for i in range(len(particles)):
                self.nodes.momentum[mapped_nodes[i, 0], 0, 0] += (
                    particles.mass[i]
                    * particles.velocity[i, 0, 0]
                    * mapped_positions[i, 0, 0]
                )
                self.nodes.momentum[mapped_nodes[i, 1], 0, 0] += (
                    particles.masss[i]
                    * particles.velocity[i, 0, 0]
                    * mapped_positions[i, 1, 0]
                )

        fill_momentum()

    def compute_nodal_velocity(self, particles):
        r"""
        Compute the nodal velocity based on particle velocity.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`v_i = \sum_p N_i(x_p) v_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()

        @ti.kernel
        def fill_velocity():
            for i in range(len(particles)):
                self.nodes.velocity[mapped_nodes[i, 0], 0, 0] += (
                    (mapped_positions[i, 0, 0] / self.nodes.mass[i, 0, 0])
                    * particles.velocity[i, 0, 0]
                    * particles.mass[i, 0, 0]
                )
                self.nodes.velocity[mapped_nodes[i, 1], 0, 0] += (
                    (mapped_positions[i, 1, 0] / self.nodes.mass[i, 0, 0])
                    * particles.velocity[i, 0, 0]
                    * particles.mass[i, 0, 0]
                )

        fill_velocity()

    def compute_external_force(self, particles):
        r"""
        Update the nodal external force based on particle f_ext.

        The nodal force is updated as a sum of particle external
        force for all particles mapped to the node.

        :math:`(f_{ext})_i = \sum_p N_i(x_p) f_{ext}`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()
        #Testing of all the functions has to be done
        @ti.kernel
        def fill_external_force():
            for i in range(len(particles)):
                self.nodes.f_ext[mapped_nodes[i, 0], 0, 0] += (
                    mapped_positions[i, 0, 0] * particles.f_ext[i, 0, 0]
                )
                self.nodes.external_force[mapped_nodes[i, 1], 0, 0] += (
                    mapped_positions[i, 1, 0] * particles.f_ext[i, 0, 0]
                )

        fill_external_force()

    def compute_body_force(self, particles, gravity: float):
        r"""
        Update the nodal external force based on particle mass.

        The nodal force is updated as a sum of particle body
        force for all particles mapped to th

        :math:`(f_{b})_i = \sum_p N_i(x_p) m_p g`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()

        @ti.kernel
        def fill_body_force():
            for i in range(len(particles)):
                self.nodes.f_ext[mapped_nodes[i, 0], 0, 0] += (
                    mapped_positions[i, 0, 0] * particles.mass[i, 0, 0] * gravity
                )
                self.nodes.f_ext[mapped_nodes[i, 1], 0, 0] += (
                    mapped_positions[i, 1, 0] * particles.mass[i, 0, 0] * gravity
                )

        fill_body_force()

#Working on the compute internal force

    def compute_internal_force(self, particles):
        mapped_nodes = ti.field(ti.i32, shape=(len(particles.loc), 2))

        @ti.kernel
        def fill_mapped_nodes():
            for i in range(len(particles.reference_loc)):
                mapped_nodes[i, 0] = self.id_to_node_ids(particles.element_ids[i])[0]
                mapped_nodes[i, 1] = self.id_to_node_ids(particles.element_ids[i])[1]

        fill_mapped_nodes()

        mapped_coords = self.id_to_node_loc(particles.element_ids)
        mapped_grads = ti.field(ti.f32, shape=(len(particles.loc), 2, 2))



    def update_nodal_momentum(self, particles, dt: float, *args):
        """Update the nodal momentum based on total force on nodes."""
        total_force = self.nodes.get_total_force()

        @ti.kernel
        def update_momentum():
            for i in range(self.nodes.nnodes):
                self.nodes.momentum[i, 0, 0] += dt * total_force[i, 0, 0]

        update_momentum()

        @ti.kernel
        def update_acceleration():
            for i in range(self.nodes.nnodes):
                self.nodes.acceleration[i, 0, 0] = (
                    total_force[i, 0, 0] / self.nodes.mass[i, 0, 0]
                )

        update_acceleration()

        @ti.kernel
        def update_velocity():
            for i in range(self.nodes.nnodes):
                self.nodes.velocity[i, 0, 0] += dt * self.nodes.acceleration[i, 0, 0]

        update_velocity()

    def apply_boundary_constraints(self):
        """Apply boundary conditions for nodal velocity."""
        @ti.kernel
        def apply_constraints():
            for i in self.boundary_nodes:
                self.nodes.velocity[i, 0, 0] = 0
                self.nodes.momentum[i, 0, 0] = 0
                self.nodes.acceleration[i, 0, 0] = 0

        apply_constraints()

    def apply_force_boundary_constraints(self):
        """Apply boundary conditions for nodal forces."""
        @ti.kernel
        def apply_constraints():
            for i in self.boundary_nodes:
                self.nodes.f_ext[i, 0, 0] = 0
                self.nodes.f_int[i, 0, 0] = 0
                self.nodes.f_damp[i, 0, 0] = 0

        apply_constraints()
