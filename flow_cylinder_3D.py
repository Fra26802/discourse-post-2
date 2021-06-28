#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:04:14 2021

@author: gpr
"""

#USEFUL LIBRARIES---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import meshio


#USEFUL FUNCTIONS---------------------------------------------------
#tensors needed for flow equation:
def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


#SIMULATION PARAMETERS--------------------------------------------------------------------
T = 10.0           # final time
num_steps = 10   # number of time steps
dt = T / num_steps # time step size


#FROM MSH FILE TO XDMF--------------------------------------
# source: J.Dokken webpage https://jsdokken.com/converted_files/tutorial_pygmsh.html------------------------------------------------
from gmsh_interface import create_mesh

mesh3D_from_msh = meshio.read("cylinder.msh")
tetra_mesh = create_mesh(mesh3D_from_msh, "tetra")
meshio.write("mesh3D_from_msh.xdmf", tetra_mesh)

triangle_mesh = create_mesh(mesh3D_from_msh, "triangle")
meshio.write("facet_mesh3D_from_msh.xdmf", triangle_mesh)


#READING THE XDMF FILE------------------------------------------------------------------------------------------------------------
#source https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/28 ----
from fenics import *

mesh3D_from_msh = Mesh()
with XDMFFile("mesh3D_from_msh.xdmf") as infile:
    infile.read(mesh3D_from_msh)
mvc = MeshValueCollection("size_t", mesh3D_from_msh, 2) 


with XDMFFile("facet_mesh3D_from_msh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh3D_from_msh, mvc)

mvc = MeshValueCollection("size_t", mesh3D_from_msh, 3)
with XDMFFile("mesh3D_from_msh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh3D_from_msh, mvc)

print('surface marker check:')
print(mf.array(), min(mf.array()), max(mf.array()))
print('volume marker check:')
print(cf.array(), min(cf.array()), max(cf.array()))


#DEFINING INTEGRATION MEASURES-------------------------------------------------------------------
dx = Measure('dx', subdomain_data=cf, domain=mesh3D_from_msh)
ds = Measure('ds', subdomain_data=mf, domain=mesh3D_from_msh)


#PHYSICAL PROPERTIES----------------------------------------------------------------------------
mu=Constant(1.)
rho=Constant(1.)


#TRIAL FUNCTIONS, TEST FUNCTIONS AND FUNCTION SPACES---------------------------
#CFD functions
V = VectorFunctionSpace(mesh3D_from_msh, 'P', 2)
Q = FunctionSpace(mesh3D_from_msh, 'P', 1)


#BOUNDARY CONDITIONS ----------------------------------------------------------
bcu_noslip  = DirichletBC(V, Constant((0, 0, 0)), mf, 11)
bcp_inflow  = DirichletBC(Q, Constant(18.6), mf, 22)
bcp_outflow = DirichletBC(Q, Constant(0), mf, 33)

bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]


#FUNCTIONS DEFINITION-----------------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

U   = 0.5 * (u_n + u)
n   = FacetNormal(mesh3D_from_msh)
f   = Constant((0, 0, 0))
k   = Constant(dt)


#NAVIER-STOKES SOLVING: IPCS IMPLEMENTATION---------------------------------------------------------------------------
#Source :fenics tutorial demo ft07: https://github.com/gdmcbain/fenics-tuto-in-skfem/commit/3d4803e2fd61a7672cd7d1a4fb4f4ccfedde6084


# Define variational problem for step 1
F1 =rho*dot((u - u_n) / k, v)*dx + \
   rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx  \
    + inner(sigma(U, p_n), epsilon(v))*dx  \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx 
a1 = lhs(F1)
L1 = rhs(F1)


a2 = dot(nabla_grad(p), nabla_grad(q))*dx 
L2 =dot(nabla_grad(p_n), nabla_grad(q))*dx  - (1/k)*div(u_)*q*dx 
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx 

A1 = assemble(a1, keep_diagonal=True)
A1.ident_zeros()

A2 = assemble(a2, keep_diagonal=True)
A2.ident_zeros()

A3 = assemble(a3, keep_diagonal=True)
A3.ident_zeros()

[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]


t = 0
for n in range(num_steps):

    t += dt

    b1 = assemble(L1)           # Step 1: Tentative velocity step
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)
    print(min(u_.vector().get_local()), max(u_.vector().get_local()))
    
    b2 = assemble(L2)           # Step 2: Pressure correction step
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)
    print(min(u_.vector().get_local()), max(u_.vector().get_local()))
    
    b3 = assemble(L3)           # Step 3: Velocity correction step
    solve(A3, u_.vector(), b3)
    print(min(u_.vector().get_local()), max(u_.vector().get_local()))
    
    u_n.assign(u_)
    p_n.assign(p_)
    

#SAVE VELOCITY FIELD FOR PARAVIEW ------------------------------------------
xdmffile_u = XDMFFile('velocity_field.xdmf')
xdmffile_u.write(u_n)

#RAPID PLOT ---------------------------------------------------------------
vit=plot(u_)
plt.colorbar (vit)
plt.show()