from dolfinx import mesh, default_scalar_type
from dolfinx.fem import form, assemble_scalar
import matplotlib.ticker as mticker
import ufl
import numpy as np
from matplotlib import pyplot as plt
from create_geometry import *
from solve_full_model import *
from draw_convergence_triangles import *

mesh_size = 1 / 64
num_refinements = 6
final_time = 1
num_steps = np.array([2**i for i in range(num_refinements)])
time_step_sizes = final_time / num_steps

u1_sols = []
u2_sols = []
p1_sols = []
p2_sols = []
pc_sols = []
mesh, subdomains, boundaries_and_interfaces = create_initial_mesh(mesh_size)

# Compute solution for a sequence of progressively smaller time steps.
for i in range(num_refinements):
    _, _, _, _, _, u1, u2, p1, p2, pc = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps[i], 1e1)
    u1_sols.append(u1)
    u2_sols.append(u2)
    p1_sols.append(p1)
    p2_sols.append(p2)
    pc_sols.append(pc)

# Compute reference solution using a very small time step.
num_steps_final = 6 * num_steps[-1]
_, _, _, _, _, u1_fine, u2_fine, p1_fine, p2_fine, pc_fine = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps_final, 1e1)
EL2u = np.zeros(len(u1_sols), dtype=default_scalar_type)
EL2p = np.zeros(len(p1_sols), dtype=default_scalar_type)
EH1u = np.zeros(len(u1_sols), dtype=default_scalar_type)
EH1p = np.zeros(len(p1_sols), dtype=default_scalar_type)
EH1pc = np.zeros(len(pc_sols), dtype=default_scalar_type)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
fracture_facets = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]
fracture_facets = ordered_interior_entities(fracture_facets, mesh, subdomains)
dS_frac = ufl.Measure(
    "dS",
    domain=mesh,
    subdomain_data=[
        (1, fracture_facets)
    ]
)
# Compute orders of convergence.
for i, (u1, u2, p1, p2, pc) in enumerate(zip(u1_sols, u2_sols, p1_sols, p2_sols, pc_sols)):
    max_err_L2_p = 0
    max_err_H1_u = 0
    error_grad_p = 0
    max_err_L2_pc = 0
    error_grad_pc = 0
    for m in range(len(u1)-1):  
        n = int((m + 1) * num_steps_final / num_steps[i])  # These are the discrete time points that the solution shares with the reference solution.
        if n / num_steps_final > 0.1:   # Do not compare the earliest time points, because of low temporal regularity of the solution.
            # Pressure H1-error
            error_L2_p = form((p1[m + 1] - p1_fine[n]) ** 2 * dx(1) + (p2[m + 1] - p2_fine[n]) ** 2 * dx(2))
            error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
            max_err_L2_p = max(max_err_L2_p, error_L2_p)
            error_grad_p_form = form(ufl.inner(ufl.grad(p1[m+1] - p1_fine[n]), ufl.grad(p1[m+1] - p1_fine[n])) * dx(1) + ufl.inner(ufl.grad(p2[m+1] - p2_fine[n]), ufl.grad(p2[m+1] - p2_fine[n])) * dx(2))
            error_grad_p += assemble_scalar(error_grad_p_form)

            # Displacement H1-error
            error_L2_u = form(ufl.dot(u1[m + 1] - u1_fine[n], u1[m + 1] - u1_fine[n]) * dx(1) + ufl.dot(u2[m + 1] - u2_fine[n], u2[m + 1] - u2_fine[n]) * dx(2))
            error_L2_u_squared = assemble_scalar(error_L2_u)
            error_grad_u = form(ufl.inner(ufl.grad(u1[m + 1] - u1_fine[n]), ufl.grad(u1[m + 1] - u1_fine[n])) * dx(1) + ufl.inner(ufl.grad(u2[m + 1] - u2_fine[n]), ufl.grad(u2[m + 1] - u2_fine[n])) * dx(2))
            error_grad_u_squared = assemble_scalar(error_grad_u)
            error_H1_u = np.sqrt(error_L2_u_squared + error_grad_u_squared)
            max_err_H1_u = max(max_err_H1_u, error_H1_u)

            # Fracture pressure H1-error
            error_L2_pc = form((pc[m + 1]("+") - pc_fine[n]("+")) ** 2 * dS_frac(1))
            error_L2_pc = np.sqrt(assemble_scalar(error_L2_pc))
            max_err_L2_pc = max(max_err_L2_pc, error_L2_pc)
            tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
            error_grad_pc_form = form(ufl.inner(ufl.dot(tang, ufl.grad(pc[m+1]("+") - pc_fine[n]("+"))), ufl.dot(tang, ufl.grad(pc[m+1]("+") - pc_fine[n]("+")))) * dS_frac(1))
            error_grad_pc += assemble_scalar(error_grad_pc_form)

    EH1u[i] = max_err_H1_u
    EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps[i] * error_grad_p)
    EH1pc[i] = max_err_L2_pc + np.sqrt(final_time / num_steps[i] * error_grad_pc)
print(f"H1-errors displacement: {EH1u}")
print(f"H1-errors pressure: {EH1p}")
print(f"H1-errors fracture pressure: {EH1pc}")
rates_H1_u = np.log(EH1u[1:] / EH1u[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
rates_H1_pc = np.log(EH1pc[1:] / EH1pc[:-1]) / np.log(num_steps[:-1] / num_steps[1:])
print(f"H1-rates displacement: {rates_H1_u}")
print(f"H1-rates pressure: {rates_H1_p}")
print(f"H1-rates fracture pressure: {rates_H1_pc}")

# Plot errors.
fig, ax = plt.subplots()
ax.loglog(num_steps, EH1u, marker='o', color='red')
ax.loglog(num_steps, EH1p, marker='o', color='blue')
ax.loglog(num_steps, EH1pc, marker='o', color='black')
ax.legend(["Displacement", "Pressure", "Fracture pressure"], fontsize=12, handlelength=3)
ax.set_xlabel(r"Number of time steps", fontsize=14)
ax.grid(which='both', alpha=0.3)
ax.set_ylabel(r"$H^1$-error", fontsize=14, labelpad=-3)
ax.tick_params(labelsize=14.5)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
draw_multiple_loglog_slopes(fig, ax, origin=(0.95 * num_steps[-1], 0.72 * EH1u[-1]), triangle_width=0.8, inverted=True, slopes=[-1.10], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.47 * num_steps[-1], 3.4 * EH1p[-1]), triangle_width=0.8, inverted=False, slopes=[-1.12], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.47 * num_steps[-1], 1.6 * EH1pc[-2]), triangle_width=0.8, inverted=False, slopes=[-1.11], labelcolor=(0.33, 0.33, 0.33))
plt.savefig("H1_errors_time.png")