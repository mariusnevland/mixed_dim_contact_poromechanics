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

mesh_size = 1 / 32
num_refinements = 6
final_time = 0.1
num_steps = np.array([2**i for i in range(num_refinements)])
time_step_sizes = final_time / num_steps

# u1_sols = []
# u2_sols = []
# p1_sols = []
# p2_sols = []
# mesh, subdomains, boundaries_and_interfaces = create_initial_nonfracture_mesh(mesh_size)

# # Compute solution for a sequence of progressively smaller time steps.
# for i in range(num_refinements):
#     _, _, _, _, u1, u2, p1, p2 = solve_biot(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps[i])
#     u1_sols.append(u1)
#     u2_sols.append(u2)
#     p1_sols.append(p1)
#     p2_sols.append(p2)

# # Compute reference solution using a very small time step.
# num_steps_final = 8 * num_steps[-1]
# _, _, _, _, u1_fine, u2_fine, p1_fine, p2_fine = solve_biot(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps_final)
# EL2u = np.zeros(len(u1_sols), dtype=default_scalar_type)
# EL2p = np.zeros(len(p1_sols), dtype=default_scalar_type)
# EH1u = np.zeros(len(u1_sols), dtype=default_scalar_type)
# EH1p = np.zeros(len(p1_sols), dtype=default_scalar_type)
# dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

# # Compute orders of convergence.
# for i, (u1, u2, p1, p2) in enumerate(zip(u1_sols, u2_sols, p1_sols, p2_sols)):
#     max_err_L2_p = 0
#     max_err_H1_u = 0
#     max_err_L2_u = 0
#     error_grad_p = 0
#     for m in range(len(u1)-1):  
#         n = int((m + 1) * num_steps_final / num_steps[i])  # These are the discrete time points that the solution shares with the reference solution.

#         # Pressure H1-error
#         error_L2_p = form((p1[m + 1] - p1_fine[n]) ** 2 * dx(1) + (p2[m + 1] - p2_fine[n]) ** 2 * dx(2))
#         error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
#         max_err_L2_p = max(max_err_L2_p, error_L2_p)
#         error_grad_p_form = form(ufl.inner(ufl.grad(p1[m+1] - p1_fine[n]), ufl.grad(p1[m+1] - p1_fine[n])) * dx(1) + ufl.inner(ufl.grad(p2[m+1] - p2_fine[n]), ufl.grad(p2[m+1] - p2_fine[n])) * dx(2))
#         error_grad_p += assemble_scalar(error_grad_p_form)

#         # Displacement H1-error
#         error_L2_u = form(ufl.dot(u1[m + 1] - u1_fine[n], u1[m + 1] - u1_fine[n]) * dx(1) + ufl.dot(u2[m + 1] - u2_fine[n], u2[m + 1] - u2_fine[n]) * dx(2))
#         error_L2_u_squared = assemble_scalar(error_L2_u)
#         error_grad_u = form(ufl.inner(ufl.grad(u1[m + 1] - u1_fine[n]), ufl.grad(u1[m + 1] - u1_fine[n])) * dx(1) + ufl.inner(ufl.grad(u2[m + 1] - u2_fine[n]), ufl.grad(u2[m + 1] - u2_fine[n])) * dx(2))
#         error_grad_u_squared = assemble_scalar(error_grad_u)
#         error_H1_u = np.sqrt(error_L2_u_squared + error_grad_u_squared)
#         max_err_H1_u = max(max_err_H1_u, error_H1_u)

#     EH1u[i] = max_err_H1_u
#     EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps[i] * error_grad_p)
# print(f"H1-errors displacement: {EH1u}")
# print(f"H1-errors pressure: {EH1p}")
# rates_H1_u = np.log(EH1u[1:] / EH1u[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
# rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
# print(f"H1-rates displacement: {rates_H1_u}")
# print(f"H1-rates pressure: {rates_H1_p}")


# Fracture flow
p1_sols = []
p2_sols = []
pc_sols = []
mesh, subdomains, boundaries_and_interfaces = create_initial_mesh(mesh_size)

# Compute solution for a sequence of progressively smaller time steps.
for i in range(num_refinements):
    _, _, _, p1, p2, pc = solve_flow_problem(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps[i])
    p1_sols.append(p1)
    p2_sols.append(p2)
    pc_sols.append(pc)

# Compute reference solution using a very small time step.
num_steps_final = 8 * num_steps[-1]
_, _, _, p1_fine, p2_fine, pc_fine = solve_flow_problem(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps_final)
# EL2u = np.zeros(len(u1_sols), dtype=default_scalar_type)
# EL2p = np.zeros(len(p1_sols), dtype=default_scalar_type)
# EH1u = np.zeros(len(u1_sols), dtype=default_scalar_type)
EH1p = np.zeros(len(p1_sols), dtype=default_scalar_type)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

# Compute orders of convergence.
for i, (p1, p2, pc) in enumerate(zip(p1_sols, p2_sols, pc_sols)):
    max_err_L2_p = 0
    max_err_H1_u = 0
    max_err_L2_u = 0
    error_grad_p = 0
    for m in range(len(p1)-1):  
        n = int((m + 1) * num_steps_final / num_steps[i])  # These are the discrete time points that the solution shares with the reference solution.

        # Pressure H1-error
        error_L2_p = form((p1[m + 1] - p1_fine[n]) ** 2 * dx(1) + (p2[m + 1] - p2_fine[n]) ** 2 * dx(2))
        error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
        max_err_L2_p = max(max_err_L2_p, error_L2_p)
        error_grad_p_form = form(ufl.inner(ufl.grad(p1[m+1] - p1_fine[n]), ufl.grad(p1[m+1] - p1_fine[n])) * dx(1) + ufl.inner(ufl.grad(p2[m+1] - p2_fine[n]), ufl.grad(p2[m+1] - p2_fine[n])) * dx(2))
        error_grad_p += assemble_scalar(error_grad_p_form)

        # Displacement H1-error
        # error_L2_u = form(ufl.dot(u1[m + 1] - u1_fine[n], u1[m + 1] - u1_fine[n]) * dx(1) + ufl.dot(u2[m + 1] - u2_fine[n], u2[m + 1] - u2_fine[n]) * dx(2))
        # error_L2_u_squared = assemble_scalar(error_L2_u)
        # error_grad_u = form(ufl.inner(ufl.grad(u1[m + 1] - u1_fine[n]), ufl.grad(u1[m + 1] - u1_fine[n])) * dx(1) + ufl.inner(ufl.grad(u2[m + 1] - u2_fine[n]), ufl.grad(u2[m + 1] - u2_fine[n])) * dx(2))
        # error_grad_u_squared = assemble_scalar(error_grad_u)
        # error_H1_u = np.sqrt(error_L2_u_squared + error_grad_u_squared)
        # max_err_H1_u = max(max_err_H1_u, error_H1_u)

    EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps[i] * error_grad_p)
print(f"H1-errors pressure: {EH1p}")
rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
print(f"H1-rates pressure: {rates_H1_p}")