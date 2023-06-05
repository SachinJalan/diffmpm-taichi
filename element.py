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
    def __init__(
        self,
        nelements: ti.i32,
        el_len: ti.f32,
        boundary_nodes: Sequence,
        nodes: Nodes = None,
    ):
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
        temp_nparr = (
            np.array([self.nodes.loc[id], self.nodes.loc[id + 1]])
            .transpose(1, 0, 2, 3)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=temp_nparr.shape)
        return f.from_numpy(temp_nparr)

    def id_node_ids(self, id: ti.i32) -> ti.field:
        temp_nparr = (
            np.array([self.nodes.loc[id], self.nodes.loc[id + 1]])
            .reshape(2, 1)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=(2, 1))
        return f.from_numpy(temp_nparr)

    def id_to_node_vel(self, id: ti.i32) -> ti.field:
        temp_nparr = (
            np.array([self.nodes.velocity[id], self.nodes.velocity[id + 1]])
            .reshape(2, 1)
            .astype(np.int32)
        )
        f = ti.field(ti.i32, shape=(2, 1))
        return f.from_numpy(temp_nparr)

    def shapefn(self, xi: float | np.ndarray):
        if len(xi.shape) != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        tmp_nparray = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)]).transpose(1, 0, 2, 3)
        result = ti.field(ti.i32, shape=tmp_nparray.shape)
        result.from_numpy(tmp_nparray)
        return result
    
    def _shapefn_natural_grad(self,xi: float | np.ndarray):
        if len(xi.shape) != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = ti.field(ti.i32, shape=(xi.shape[0],2))
        for i in range(xi.shape[0]):
            result[i,0] = -0.5
            result[i,1] = 0.5
        return result
    
    def shapefn_grad(self, xi: float | np.ndarray,coords):
