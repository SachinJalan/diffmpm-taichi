import abc
from typing import Iterable

import taichi as ti

from diffmpm.element import _Element
from diffmpm.particle import Particles

class _MeshBase(abc.ABC):
    """
    Base class for Meshes.

    Note: If attributes other than elements and particles are added
    then the child class should also implement `tree_flatten` and
    `tree_unflatten` correctly or that information will get lost.
    """

    def __init__(self, config: dict):
        """Initialize mesh using configuration."""
        self.particles: Iterable[Particles, ...] = config["particles"]
        self.elements: _Element = config["elements"]

    def apply_on_elements(self,function,args=()):
        f = getattr(self.elements,function)
        for particle_set in self.particles:
            f(particle_set,*args)
    
    def apply_on_particles(self,function,args=()):
        for particle_set in self.particles:
            f = getattr(particle_set,function)
            f(self.elements,*args)

class Mesh1D(_MeshBase):

    def __init__(self,config:dict):
        super().__init__(config)

