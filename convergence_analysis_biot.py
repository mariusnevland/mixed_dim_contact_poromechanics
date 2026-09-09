from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form, create_interpolation_data, assemble_scalar
import matplotlib.ticker as mticker
import ufl
import numpy as np
from matplotlib import pyplot as plt
from create_geometry import *
from solve_full_model import *
from flow_testing import *
from solve_biot import *

initial_mesh_size = 1 / 4
num_grids = 6
final_time = 0.1
num_steps = 1

V1_spaces = []
V2_spaces = []
Q1_spaces = []
Q2_spaces = []
u1_sols = []
u2_sols = []
p1_sols = []
p2_sols = []
cells_data = []
hs = np.zeros(num_grids-1, dtype=default_scalar_type)
mesh, subdomains, boundaries_and_interfaces = create_initial_nonfracture_mesh(initial_mesh_size)

# Flow problem
for i in range(num_grids-1):
    V1, V2, Q1, Q2, u1, u2, p1, p2 = solve_biot(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
    V1_spaces.append(V1)
    V2_spaces.append(V2)
    Q1_spaces.append(Q1)
    Q2_spaces.append(Q2)
    u1_sols.append(u1)
    u2_sols.append(u2)
    p1_sols.append(p1)
    p2_sols.append(p2)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    cells_data.append(cells)
    h = np.max(mesh.h(mesh.topology.dim, cells))
    hs[i] = h
    if not i==num_grids-2:
        mesh, subdomains, boundaries_and_interfaces = refine_mesh_with_tags(mesh, subdomains, boundaries_and_interfaces)


# Compute solution on very fine mesh as the reference solution.
for _ in range(2):
    mesh, subdomains, boundaries_and_interfaces = refine_mesh_with_tags(mesh, subdomains, boundaries_and_interfaces)
V1_fine, V2_fine, Q1_fine, Q2_fine, u1_fine, u2_fine, p1_fine, p2_fine = solve_biot(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells_fine = np.arange(num_cells, dtype=np.int32)
dx_fine = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

# Compute orders of convergence.
EH1p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1u = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
for i, (V1, V2, Q1, Q2, u1, u2, p1, p2, cells) in enumerate(zip(V1_spaces, V2_spaces, Q1_spaces, Q2_spaces, u1_sols, u2_sols, p1_sols, p2_sols, cells_data)):  # Loop through all grid sizes
    max_err_L2_p = 0
    max_err_H1_u = 0
    error_grad_p = 0
    max_err_L2_pc = 0
    error_grad_pc = 0
    interp_data_p1 = create_interpolation_data(Q1_fine, Q1, cells_fine)
    interp_data_p2 = create_interpolation_data(Q2_fine, Q2, cells_fine)
    interp_data_u1 = create_interpolation_data(V1_fine, V1, cells_fine)
    interp_data_u2 = create_interpolation_data(V2_fine, V2, cells_fine)
    for (u1t, u2t, p1t, p2t, u1fine_t, u2fine_t, p1fine_t, p2fine_t) in zip(u1, u2, p1, p2, u1_fine, u2_fine, p1_fine, p2_fine):  # Loop through all time steps
        # Interpolate current solution onto the reference solution mesh.
        p1_fine_interp = fem.Function(Q1_fine)
        p2_fine_interp = fem.Function(Q2_fine)
        u1_fine_interp = fem.Function(V1_fine)
        u2_fine_interp = fem.Function(V2_fine)
        p1_fine_interp.interpolate_nonmatching(p1t, cells_fine, interpolation_data=interp_data_p1)
        p2_fine_interp.interpolate_nonmatching(p2t, cells_fine, interpolation_data=interp_data_p2)
        u1_fine_interp.interpolate_nonmatching(u1t, cells_fine, interpolation_data=interp_data_u1)
        u2_fine_interp.interpolate_nonmatching(u2t, cells_fine, interpolation_data=interp_data_u2)
        # Pressure H1-error
        error_L2_p = form((p1fine_t - p1_fine_interp) ** 2 * dx_fine(1) + (p2fine_t - p2_fine_interp) ** 2 * dx_fine(2))
        error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
        max_err_L2_p = max(max_err_L2_p, error_L2_p)
        error_grad_p_form = form(ufl.inner(ufl.grad(p1fine_t - p1_fine_interp), ufl.grad(p1fine_t - p1_fine_interp)) * dx_fine(1) + ufl.inner(ufl.grad(p2fine_t - p2_fine_interp), ufl.grad(p2fine_t - p2_fine_interp)) * dx_fine(2))          
        error_grad_p += assemble_scalar(error_grad_p_form)
        # Displacement H1-error
        error_L2_u = form(ufl.dot(u1fine_t - u1_fine_interp, u1fine_t - u1_fine_interp) * dx_fine(1) + ufl.dot(u2fine_t - u2_fine_interp, u2fine_t - u2_fine_interp) * dx_fine(2))
        error_L2_u_squared = assemble_scalar(error_L2_u)
        error_grad_u = form(ufl.inner(ufl.grad(u1fine_t - u1_fine_interp), ufl.grad(u1fine_t - u1_fine_interp)) * dx_fine(1) + ufl.inner(ufl.grad(u2fine_t - u2_fine_interp), ufl.grad(u2fine_t - u2_fine_interp)) * dx_fine(2))
        error_grad_u_squared = assemble_scalar(error_grad_u)
        error_H1_u = np.sqrt(error_L2_u_squared + error_grad_u_squared)
        max_err_H1_u = max(max_err_H1_u, error_H1_u)

    EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps * error_grad_p)
    EH1u[i] = max_err_H1_u
print(f"H1-errors displacement: {EH1u}")
print(f"H1-errors pressure: {EH1p}")
rates_H1_u = np.log(EH1u[1:] / EH1u[:-1]) / np.log(hs[1:] / hs[:-1])
rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(hs[1:] / hs[:-1])
print(f"H1-rates displacement: {rates_H1_u}")
print(f"H1-rates pressure: {rates_H1_p}")