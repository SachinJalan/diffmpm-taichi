import abc
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


@ti.data_oriented
class Material(abc.ABC):
    _props = ()

    def __init__(self, material_properties):
        self.properties = material_properties

    @abc.abstractmethod
    def __repr__(self):
        ...

    @abc.abstractmethod
    def compute_stress(self):
        ...

    def validate_props(self, material_properties):
        for key in self._props:
            if key not in material_properties:
                raise KeyError(
                    f"'{key}' should be present in `material_properties` for {self.__class__.__name__} materials."
                )


@ti.data_oriented
class LinearElastic(Material):
    _props = ("density", "youngs_modulus", "poisson_ratio")

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        youngs_modulus = material_properties["youngs_modulus"]
        poisson_ratio = material_properties["poisson_ratio"]
        density = material_properties["density"]
        bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
        constrained_modulus = (
            youngs_modulus
            * (1 - poisson_ratio)
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
        # Wave velocities
        vp = tm.sqrt(constrained_modulus / density)
        vs = tm.sqrt(shear_modulus / density)
        self.properties = {
            **material_properties,
            "bulk_modulus": bulk_modulus,
            "pwave_velocity": vp,
            "swave_velocity": vs,
        }
        self._compute_elastic_tensor()

    def __repr__(self):
        return f"LinearElastic(props={self.properties})"

    def _compute_elastic_tensor(self):
        G = self.properties["youngs_modulus"] / (
            2 * (1 + self.properties["poisson_ratio"])
        )

        a1 = self.properties["bulk_modulus"] + (4 * G / 3)
        a2 = self.properties["bulk_modulus"] - (2 * G / 3)

        self.de = ti.Matrix(
            [
                [a1, a2, a2, 0, 0, 0],
                [a2, a1, a2, 0, 0, 0],
                [a2, a2, a1, 0, 0, 0],
                [0, 0, 0, G, 0, 0],
                [0, 0, 0, 0, G, 0],
                [0, 0, 0, 0, 0, G],
            ]
        )

    def compute_stress(self, dstrain: ti.Matrix) -> ti.Matrix:
        """
        Compute material stress.
        """
        dstress = self.de@dstrain
        return dstress


@ti.data_oriented
class SimpleMaterial(Material):
    _props = ("E", "density")

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        self.properties = material_properties

    def __repr__(self):
        return f"SimpleMaterial(props={self.properties})"

    def compute_stress(self, dstrain: ti.Matrix) -> ti.Matrix:
        return dstrain * self.properties["E"]
