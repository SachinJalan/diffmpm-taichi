import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
ti.init(arch=ti.gpu)

nsteps = 5200
L = 25
rho = 1
nelements = 13
dx = L / nelements
x_n = ti.field(ti.f32, shape=(nelements + 1))


@ti.kernel
def fillx_n():
    for i in range(nelements + 1):
        x_n[i] = i * dx


fillx_n()
nnodes = nelements + 1
elements = ti.field(ti.i32, shape=(nelements, 2))
elements.fill(0)


@ti.kernel
def fillelements():
    for i in range(nelements):
        elements[i, 0] = i
        elements[i, 1] = i + 1


fillelements()
v0 = 0.1
E = 100
c = tm.sqrt(E / rho)
b1 = tm.pi / (2 * L)
w1 = b1 * c
nparticles = nelements
pmid = 6
x_p = ti.field(ti.f32, shape=(nparticles))
x_p.fill(0)
vol_p = ti.field(ti.f32, shape=(nparticles))
vol_p.fill(1)
mass_p = ti.field(ti.f32, shape=(nparticles))
mass_p.fill(rho)
stress_p = ti.field(ti.f32, shape=(nparticles))
stress_p.fill(0)
vel_p = ti.field(ti.f32, shape=(nparticles))
vel_p.fill(0)


@ti.kernel
def fillx_p_v_p():
    for i in range(nparticles):
        x_p[i] = 0.5 * (x_n[i] + x_n[i + 1])
        vel_p[i] = v0 * tm.sin(b1 * x_p[i])


dt_crit = dx / c
dt = 0.02
fillx_p_v_p()

vt = ti.field(ti.f32, shape=nsteps)
tt = ti.field(ti.f32, shape=nsteps)
vt.fill(0)
tt.fill(0)
xt = ti.field(ti.f32, shape=nsteps)
xt.fill(0)

mass_n = ti.field(ti.f32, shape=nnodes)
mass_n.fill(0)
mom_n = ti.field(ti.f32, shape=nnodes)
mom_n.fill(0)
fint_n = ti.field(ti.f32, shape=nnodes)
fint_n.fill(0)


@ti.kernel
def step(i:ti.i32):
    # ti.loop_config(serialize=True)
    # for i in range(nsteps):
        # reset nodal values
    for j in range(nnodes):
        mass_n[j] = 0
        mom_n[j] = 0
        fint_n[j] = 0
    for eid in range(nelements):
        nid1, nid2 = elements[eid,0],elements[eid,1]

        N1 = 1 - abs(x_p[eid] - x_n[nid1]) / dx
        N2 = 1 - abs(x_p[eid] - x_n[nid2]) / dx
        dN1 = -1 / dx
        dN2 = 1 / dx

        mass_n[nid1] += N1 * mass_p[eid]
        mass_n[nid2] += N2 * mass_p[eid]

        mom_n[nid1] += N1 * mass_p[eid] * vel_p[eid]
        mom_n[nid2] += N2 * mass_p[eid] * vel_p[eid]

        fint_n[nid1] -= vol_p[eid] * stress_p[eid] * dN1
        fint_n[nid2] -= vol_p[eid] * stress_p[eid] * dN2
    mom_n[0] = 0
    fint_n[0] = 0
    for nodes in range(nnodes):
        mom_n[nodes] += fint_n[nodes] * dt
    for eid in range(nelements):
        nid1, nid2 = elements[eid,0],elements[eid,1]

        N1 = 1 - abs(x_p[eid] - x_n[nid1]) / dx
        N2 = 1 - abs(x_p[eid] - x_n[nid2]) / dx

        dN1 = -1 / dx
        dN2 = 1 / dx
        vel_p[eid] += dt * N1 * fint_n[nid1] / mass_n[nid1]
        vel_p[eid] += dt * N1 * fint_n[nid2] / mass_n[nid2]

        x_p[eid] += dt * (
            (N1 * mom_n[nid1] / mass_n[nid1]) + (N2 * mom_n[nid2] / mass_n[nid2])
        )

        nv1=mom_n[nid1]/mass_n[nid1]
        nv2=mom_n[nid2]/mass_n[nid2]

        grad_v=dN1*nv1 + dN2*nv2

        dstrain=grad_v *dt
        
        vol_p[eid]*=(1+dstrain)
        stress_p[eid]+=E*dstrain
    vt[i]=vel_p[pmid]
    xt[i]=x_p[pmid]

for i in range(nsteps):
    step(i)

@ti.kernel
def filltt():
    for i in range(nsteps):
        tt[i]=i*dt
filltt()

tt_np=tt.to_numpy()
vt_np=vt.to_numpy()

plt.plot(tt,vt)
plt.show()