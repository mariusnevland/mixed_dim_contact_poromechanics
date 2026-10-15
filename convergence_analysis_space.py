from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form, create_interpolation_data, assemble_scalar
import matplotlib.ticker as mticker
import ufl
import numpy as np
from matplotlib import pyplot as plt
from utility_functions import *
from solve_full_model import *

initial_mesh_size = 1 / 6
num_grids = 6
final_time = 0.01
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
    V1, V2, Q1, Q2, Qc, u1, u2, p1, p2, pc = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps, 1e2)
    V1_spaces.append(V1)
    V2_spaces.append(V2)
    Q1_spaces.append(Q1)
    Q2_spaces.append(Q2)
    Qc_spaces.append(Qc)
    u1_sols.append(u1)
    u2_sols.append(u2)
    p1_sols.append(p1)
    p2_sols.append(p2)
    pc_sols.append(pc)
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
V1_fine, V2_fine, Q1_fine, Q2_fine, Qc_fine, u1_fine, u2_fine, p1_fine, p2_fine, pc_fine = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps, 1e2, plot_sol=True)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells_fine = np.arange(num_cells, dtype=np.int32)
dx_fine = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
fracture_facets = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]
fracture_facets = ordered_interior_entities(fracture_facets, mesh, subdomains)
dS_frac = ufl.Measure(
    "dS",
    domain=mesh,
    subdomain_data=[
        (1, fracture_facets)
    ]
)
dx_away, dS_frac_away = measure_away_from_tips(mesh, subdomains, boundaries_and_interfaces)

# Compute orders of convergence.
EL2p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1pc = np.zeros(len(Qc_spaces), dtype=default_scalar_type)
EH1u = np.zeros(len(V1_spaces), dtype=default_scalar_type)
EL2p_no_tips = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1p_no_tips = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1pc_no_tips = np.zeros(len(Qc_spaces), dtype=default_scalar_type)
EH1u_no_tips = np.zeros(len(V1_spaces), dtype=default_scalar_type)
u1_fine = u1_fine[-1]
u2_fine = u2_fine[-1]
p1_fine = p1_fine[-1]
p2_fine = p2_fine[-1]
pc_fine = pc_fine[-1]
for i, (V1, V2, Q1, Q2, Qc, u1, u2, p1, p2, pc, cells) in enumerate(zip(V1_spaces, V2_spaces, Q1_spaces, Q2_spaces, Qc_spaces, u1_sols, u2_sols, p1_sols, p2_sols, pc_sols, cells_data)):  # Loop through all grid sizes
    interp_data_p1 = create_interpolation_data(Q1_fine, Q1, cells_fine)
    interp_data_p2 = create_interpolation_data(Q2_fine, Q2, cells_fine)
    interp_data_pc = create_interpolation_data(Qc_fine, Qc, cells_fine)
    interp_data_u1 = create_interpolation_data(V1_fine, V1, cells_fine)
    interp_data_u2 = create_interpolation_data(V2_fine, V2, cells_fine)
    # Interpolate current solution onto the reference solution mesh.
    p1_fine_interp = fem.Function(Q1_fine)
    p2_fine_interp = fem.Function(Q2_fine)
    pc_fine_interp = fem.Function(Qc_fine)
    u1_fine_interp = fem.Function(V1_fine)
    u2_fine_interp = fem.Function(V2_fine)
    p1_fine_interp.interpolate_nonmatching(p1[-1], cells_fine, interpolation_data=interp_data_p1)
    p2_fine_interp.interpolate_nonmatching(p2[-1], cells_fine, interpolation_data=interp_data_p2)
    pc_fine_interp.interpolate_nonmatching(pc[-1], cells_fine, interpolation_data=interp_data_pc)
    u1_fine_interp.interpolate_nonmatching(u1[-1], cells_fine, interpolation_data=interp_data_u1)
    u2_fine_interp.interpolate_nonmatching(u2[-1], cells_fine, interpolation_data=interp_data_u2)

    # Pressure H1-error
    error_L2_p = form((p1_fine - p1_fine_interp) ** 2 * dx_fine(1) + (p2_fine - p2_fine_interp) ** 2 * dx_fine(2))
    error_L2_p = assemble_scalar(error_L2_p)
    error_grad_p = form(ufl.inner(ufl.grad(p1_fine - p1_fine_interp), ufl.grad(p1_fine - p1_fine_interp)) * dx_fine(1) + ufl.inner(ufl.grad(p2_fine - p2_fine_interp), ufl.grad(p2_fine - p2_fine_interp)) * dx_fine(2))
    error_grad_p = assemble_scalar(error_grad_p)

    # Fracture pressure H1-error
    error_L2_pc = form((pc_fine("+") - pc_fine_interp("+")) ** 2 * dS_frac(1))
    error_L2_pc = assemble_scalar(error_L2_pc)
    tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the horizontal fracture, used to compute the tangential gradient along the fracture.
    error_grad_pc = form(ufl.inner(ufl.dot(tang, ufl.grad(pc_fine("+") - pc_fine_interp("+"))), ufl.dot(tang, ufl.grad(pc_fine("+") - pc_fine_interp("+")))) * dS_frac(1))
    error_grad_pc = assemble_scalar(error_grad_pc)

    # Displacement H1-error
    error_L2_u = form(ufl.dot(u1_fine - u1_fine_interp, u1_fine - u1_fine_interp) * dx_fine(1) + ufl.dot(u2_fine - u2_fine_interp, u2_fine - u2_fine_interp) * dx_fine(2))
    error_L2_u = assemble_scalar(error_L2_u)
    error_grad_u = form(ufl.inner(ufl.grad(u1_fine - u1_fine_interp), ufl.grad(u1_fine - u1_fine_interp)) * dx_fine(1) + ufl.inner(ufl.grad(u2_fine - u2_fine_interp), ufl.grad(u2_fine - u2_fine_interp)) * dx_fine(2))
    error_grad_u = assemble_scalar(error_grad_u)

    EH1p[i] = np.sqrt(error_L2_p) + np.sqrt(final_time / num_steps * error_grad_p)
    EH1pc[i] = np.sqrt(error_L2_pc) + np.sqrt(final_time / num_steps * error_grad_pc)
    EH1u[i] = np.sqrt(error_L2_u + error_grad_u)

    ### Recompute the errors, but with a small radius around each fracture tip removed, to test the effect of fracture tip singularities on the convergence rate.
    # Pressure H1-error
    error_L2_p_no_tips = form((p1_fine - p1_fine_interp) ** 2 * dx_away(1) + (p2_fine - p2_fine_interp) ** 2 * dx_away(2))
    error_L2_p_no_tips = assemble_scalar(error_L2_p_no_tips)
    error_grad_p_no_tips = form(ufl.inner(ufl.grad(p1_fine - p1_fine_interp), ufl.grad(p1_fine - p1_fine_interp)) * dx_away(1) + ufl.inner(ufl.grad(p2_fine - p2_fine_interp), ufl.grad(p2_fine - p2_fine_interp)) * dx_away(2))
    error_grad_p_no_tips = assemble_scalar(error_grad_p_no_tips)

    # Fracture pressure H1-error
    error_L2_pc_no_tips = form((pc_fine("+") - pc_fine_interp("+")) ** 2 * dS_frac_away(1))
    error_L2_pc_no_tips = assemble_scalar(error_L2_pc_no_tips)
    error_grad_pc_no_tips = form(ufl.inner(ufl.dot(tang, ufl.grad(pc_fine("+") - pc_fine_interp("+"))), ufl.dot(tang, ufl.grad(pc_fine("+") - pc_fine_interp("+")))) * dS_frac_away(1))
    error_grad_pc_no_tips = assemble_scalar(error_grad_pc_no_tips)

    # Displacement H1-error
    error_L2_u_no_tips = form(ufl.dot(u1_fine - u1_fine_interp, u1_fine - u1_fine_interp) * dx_away(1) + ufl.dot(u2_fine - u2_fine_interp, u2_fine - u2_fine_interp) * dx_away(2))
    error_L2_u_no_tips = assemble_scalar(error_L2_u_no_tips)
    error_grad_u_no_tips = form(ufl.inner(ufl.grad(u1_fine - u1_fine_interp), ufl.grad(u1_fine - u1_fine_interp)) * dx_away(1) + ufl.inner(ufl.grad(u2_fine - u2_fine_interp), ufl.grad(u2_fine - u2_fine_interp)) * dx_away(2))
    error_grad_u_no_tips = assemble_scalar(error_grad_u_no_tips)

    EH1p_no_tips[i] = np.sqrt(error_L2_p_no_tips) + np.sqrt(final_time / num_steps * error_grad_p_no_tips)
    EH1pc_no_tips[i] = np.sqrt(error_L2_pc_no_tips) + np.sqrt(final_time / num_steps * error_grad_pc_no_tips)
    EH1u_no_tips[i] = np.sqrt(error_L2_u_no_tips + error_grad_u_no_tips)

# Print errors and convergence rates with fracture tips included.
print("--------------------------------")
print("Errors and convergence rates with fracture tips included:")
print("--------------------------------")
print(f"Errors displacement: {EH1u}")
print(f"Errors pressure: {EH1p}")
print(f"Errors fracture pressure: {EH1pc}")
rates_u = np.log(EH1u[1:] / EH1u[:-1]) / np.log(hs[1:] / hs[:-1])
rates_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(hs[1:] / hs[:-1])
rates_pc = np.log(EH1pc[1:] / EH1pc[:-1]) / np.log(hs[1:] / hs[:-1])
print(f"Rates displacement: {rates_u}")
print(f"Rates pressure: {rates_p}")
print(f"Rates fracture pressure: {rates_pc}")

# Print errors and convergence rates with fracture tips removed.
print("--------------------------------")
print("Errors and convergence rates with fracture tips removed:")
print("--------------------------------")
print(f"Errors displacement: {EH1u_no_tips}")
print(f"Errors pressure: {EH1p_no_tips}")
print(f"Errors fracture pressure: {EH1pc_no_tips}")
rates_u_no_tips = np.log(EH1u_no_tips[1:] / EH1u_no_tips[:-1]) / np.log(hs[1:] / hs[:-1])
rates_p_no_tips = np.log(EH1p_no_tips[1:] / EH1p_no_tips[:-1]) / np.log(hs[1:] / hs[:-1])
rates_pc_no_tips = np.log(EH1pc_no_tips[1:] / EH1pc_no_tips[:-1]) / np.log(hs[1:] / hs[:-1])
print(f"Rates displacement: {rates_u_no_tips}")
print(f"Rates pressure: {rates_p_no_tips}")
print(f"Rates fracture pressure: {rates_pc_no_tips}")

# Plot errors.
hs_inv = 1 / hs
fig, ax = plt.subplots()
ax.loglog(hs_inv, EH1u, marker='o', color='red')
ax.loglog(hs_inv, EH1p, marker='o', color='blue')
ax.loglog(hs_inv, EH1pc, marker='o', color='black')
ax.legend(["Displacement", "Pressure", "Fracture pressure"], fontsize=11, handlelength=3, loc="upper left", bbox_to_anchor=(0.05, 0.5))
ax.set_xlabel("Inverse of mesh size (1/h)", fontsize=14)
ax.grid(which='both', alpha=0.3)
ax.set_ylabel("Error", fontsize=14, labelpad=-3)
ax.set_xticks([5, 1e1, 2e1, 4e1, 8e1])
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.tick_params(labelsize=14.5)
draw_multiple_loglog_slopes(fig, ax, origin=(0.95 * hs_inv[-1], 0.76 * EH1u[-1]), triangle_width=1, inverted=True, slopes=[-0.68], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.48 * hs_inv[-1], 1.58 * EH1p[-2]), triangle_width=1, inverted=False, slopes=[-0.86], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.95 * hs_inv[-1], 0.76 * EH1pc[-1]), triangle_width=1, inverted=True, slopes=[-0.43], labelcolor=(0.33, 0.33, 0.33))
plt.savefig("errors_space.png")
fig, ax = plt.subplots()
ax.loglog(hs_inv, EH1u_no_tips, marker='o', color='red')
ax.loglog(hs_inv, EH1p_no_tips, marker='o', color='blue')
ax.loglog(hs_inv, EH1pc_no_tips, marker='o', color='black')
ax.legend(["Displacement", "Pressure", "Fracture pressure"], fontsize=11, handlelength=3, loc="upper left", bbox_to_anchor=(0.05, 0.5))
ax.set_xlabel("Inverse of mesh size (1/h)", fontsize=14)
ax.grid(which='both', alpha=0.3)
ax.set_ylabel("Error", fontsize=14, labelpad=-3)
ax.set_xticks([5, 1e1, 2e1, 4e1, 8e1])
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.tick_params(labelsize=14.5)
draw_multiple_loglog_slopes(fig, ax, origin=(0.95 * hs_inv[-1], 0.76 * EH1u_no_tips[-1]), triangle_width=1, inverted=True, slopes=[-0.87], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.48 * hs_inv[-1], 1.5 * EH1p_no_tips[-2]), triangle_width=1, inverted=False, slopes=[-1.17], labelcolor=(0.33, 0.33, 0.33))
draw_multiple_loglog_slopes(fig, ax, origin=(0.95 * hs_inv[-1], 0.76 * EH1pc_no_tips[-1]), triangle_width=1, inverted=True, slopes=[-1.17], labelcolor=(0.33, 0.33, 0.33))   
plt.savefig("errors_space_no_fracture_tips.png")