from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form, create_interpolation_data, assemble_scalar
import matplotlib.ticker as mticker
import ufl
import numpy as np
from matplotlib import pyplot as plt
from create_geometry import *
from solve_full_model import *
from flow_testing import *

initial_mesh_size = 1 / 16
num_grids = 2
final_time = 0.1
num_steps = 1

V1_spaces = []
V2_spaces = []
Q1_spaces = []
Q2_spaces = []
Qc_spaces = []
u1_sols = []
u2_sols = []
p1_sols = []
p2_sols = []
pc_sols = []
cells_data = []
hs = np.zeros(num_grids-1, dtype=default_scalar_type)
mesh, subdomains, boundaries_and_interfaces = create_initial_mesh(initial_mesh_size)

# Compute solution on a sequence of nested, refined meshes.
for i in range(num_grids-1):
    _ = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps, friction=True)
    mesh, subdomains, boundaries_and_interfaces = refine_mesh_with_tags(mesh, subdomains, boundaries_and_interfaces)

# # Flow problem
# for i in range(num_grids-1):
#     Q1, Q2, Qc, p1, p2, pc = solve_flow_problem(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
#     Q1_spaces.append(Q1)
#     Q2_spaces.append(Q2)
#     Qc_spaces.append(Qc)
#     p1_sols.append(p1)
#     p2_sols.append(p2)
#     pc_sols.append(pc)
#     num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
#     cells = np.arange(num_cells, dtype=np.int32)
#     cells_data.append(cells)
#     h = np.max(mesh.h(mesh.topology.dim, cells))
#     hs[i] = h
#     if not i==num_grids-2:
#         mesh, subdomains, boundaries_and_interfaces = refine_mesh_with_tags(mesh, subdomains, boundaries_and_interfaces)


# # Compute solution on very fine mesh as the reference solution.
# for _ in range(2):
#     mesh, subdomains, boundaries_and_interfaces = refine_mesh_with_tags(mesh, subdomains, boundaries_and_interfaces)
# Q1_fine, Q2_fine, Qc_fine, p1_fine, p2_fine, pc_fine = solve_flow_problem(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
# num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
# cells_fine = np.arange(num_cells, dtype=np.int32)
# dx_fine = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
# ds_fine = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries_and_interfaces)

# # Compute orders of convergence.
# EL2p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
# EH1p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
# EH1pc = np.zeros(len(Qc_spaces), dtype=default_scalar_type)
# for i, (Q1, Q2, Qc, p1, p2, pc, cells) in enumerate(zip(Q1_spaces, Q2_spaces, Qc_spaces, p1_sols, p2_sols, pc_sols, cells_data)):  # Loop through all grid sizes
#     max_err_L2_p = 0
#     max_err_H1_u = 0
#     error_grad_p = 0
#     max_err_L2_pc = 0
#     error_grad_pc = 0
#     interp_data_p1 = create_interpolation_data(Q1_fine, Q1, cells_fine)
#     interp_data_p2 = create_interpolation_data(Q2_fine, Q2, cells_fine)
#     interp_data_pc = create_interpolation_data(Qc_fine, Qc, cells_fine)
#     for (p1t, p2t, pct, p1fine_t, p2fine_t, pcfine_t) in zip(p1, p2, pc, p1_fine, p2_fine, pc_fine):  # Loop through all time steps
#         # Interpolate current solution onto the reference solution mesh.
#         p1_fine_interp = fem.Function(Q1_fine)
#         p2_fine_interp = fem.Function(Q2_fine)
#         pc_fine_interp = fem.Function(Qc_fine)
#         p1_fine_interp.interpolate_nonmatching(p1t, cells_fine, interpolation_data=interp_data_p1)
#         p2_fine_interp.interpolate_nonmatching(p2t, cells_fine, interpolation_data=interp_data_p2)
#         pc_fine_interp.interpolate_nonmatching(pct, cells_fine, interpolation_data=interp_data_pc)
#         # Pressure H1-error
#         error_L2_p = form((p1fine_t - p1_fine_interp) ** 2 * dx_fine(1) + (p2fine_t - p2_fine_interp) ** 2 * dx_fine(2))
#         error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
#         max_err_L2_p = max(max_err_L2_p, error_L2_p)
#         error_grad_p_form = form(ufl.inner(ufl.grad(p1fine_t - p1_fine_interp), ufl.grad(p1fine_t - p1_fine_interp)) * dx_fine(1) + ufl.inner(ufl.grad(p2fine_t - p2_fine_interp), ufl.grad(p2fine_t - p2_fine_interp)) * dx_fine(2))          
#         error_grad_p += assemble_scalar(error_grad_p_form)
#         # Fracture pressure H1-error
#         error_L2_pc = form((pcfine_t - pc_fine_interp) ** 2 * ds_fine(6))
#         error_L2_pc = np.sqrt(assemble_scalar(error_L2_pc))
#         max_err_L2_pc = max(max_err_L2_pc, error_L2_pc)
#         tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
#         error_grad_pc_form = form(ufl.inner(ufl.dot(tang, ufl.grad(pcfine_t - pc_fine_interp)), ufl.dot(tang, ufl.grad(pcfine_t - pc_fine_interp))) * ds_fine(6))
#         error_grad_pc += assemble_scalar(error_grad_pc_form)

#     EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps * error_grad_p)
#     EL2p[i] = max_err_L2_p
#     EH1pc[i] = max_err_L2_pc + np.sqrt(final_time / num_steps * error_grad_pc)
# print(f"H1-errors pressure: {EH1p}")
# print(f"L2-errors pressure: {EL2p}")
# print(f"H1-errors fracture pressure: {EH1pc}")
# rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(hs[1:] / hs[:-1])
# rates_L2_p = np.log(EL2p[1:] / EL2p[:-1]) / np.log(hs[1:] / hs[:-1])
# rates_H1_pc = np.log(EH1pc[1:] / EH1pc[:-1]) / np.log(hs[1:] / hs[:-1])
# print(f"H1-rates pressure: {rates_H1_p}")
# print(f"L2-rates pressure: {rates_L2_p}")
# print(f"H1-rates fracture pressure: {rates_H1_pc}")