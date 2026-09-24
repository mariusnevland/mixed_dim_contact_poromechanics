from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form, create_interpolation_data, assemble_scalar
import matplotlib.ticker as mticker
import ufl
import numpy as np
from matplotlib import pyplot as plt
from create_geometry import *
from solve_full_model import *
from flow_testing import *
from draw_convergence_triangles import *

initial_mesh_size = 1 / 4
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
    V1, V2, Q1, Q2, Qc, u1, u2, p1, p2, pc = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
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
V1_fine, V2_fine, Q1_fine, Q2_fine, Qc_fine, u1_fine, u2_fine, p1_fine, p2_fine, pc_fine = solve_full_model(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells_fine = np.arange(num_cells, dtype=np.int32)
dx_fine = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds_fine = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries_and_interfaces)
fracture_facets = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]
fracture_facets = ordered_interior_entities(fracture_facets, mesh, subdomains)
dS_frac = ufl.Measure(
    "dS",
    domain=mesh,
    subdomain_data=[
        (1, fracture_facets)
    ]
)
dx_test, dS_frac_away = measure_away_from_tips(mesh, subdomains, boundaries_and_interfaces)

# Compute orders of convergence.
EL2p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1p = np.zeros(len(Q1_spaces), dtype=default_scalar_type)
EH1pc = np.zeros(len(Qc_spaces), dtype=default_scalar_type)
EH1u = np.zeros(len(V1_spaces), dtype=default_scalar_type)
for i, (V1, V2, Q1, Q2, Qc, u1, u2, p1, p2, pc, cells) in enumerate(zip(V1_spaces, V2_spaces, Q1_spaces, Q2_spaces, Qc_spaces, u1_sols, u2_sols, p1_sols, p2_sols, pc_sols, cells_data)):  # Loop through all grid sizes
    max_err_L2_p = 0
    max_err_H1_u = 0
    error_grad_p = 0
    max_err_L2_pc = 0
    error_grad_pc = 0
    interp_data_p1 = create_interpolation_data(Q1_fine, Q1, cells_fine)
    interp_data_p2 = create_interpolation_data(Q2_fine, Q2, cells_fine)
    interp_data_pc = create_interpolation_data(Qc_fine, Qc, cells_fine)
    interp_data_u1 = create_interpolation_data(V1_fine, V1, cells_fine)
    interp_data_u2 = create_interpolation_data(V2_fine, V2, cells_fine)
    for (u1t, u2t, p1t, p2t, pct, u1fine_t, u2fine_t, p1fine_t, p2fine_t, pcfine_t) in zip(u1, u2, p1, p2, pc, u1_fine, u2_fine, p1_fine, p2_fine, pc_fine):  # Loop through all time steps
        # Interpolate current solution onto the reference solution mesh.
        p1_fine_interp = fem.Function(Q1_fine)
        p2_fine_interp = fem.Function(Q2_fine)
        pc_fine_interp = fem.Function(Qc_fine)
        u1_fine_interp = fem.Function(V1_fine)
        u2_fine_interp = fem.Function(V2_fine)
        p1_fine_interp.interpolate_nonmatching(p1t, cells_fine, interpolation_data=interp_data_p1)
        p2_fine_interp.interpolate_nonmatching(p2t, cells_fine, interpolation_data=interp_data_p2)
        pc_fine_interp.interpolate_nonmatching(pct, cells_fine, interpolation_data=interp_data_pc)
        u1_fine_interp.interpolate_nonmatching(u1t, cells_fine, interpolation_data=interp_data_u1)
        u2_fine_interp.interpolate_nonmatching(u2t, cells_fine, interpolation_data=interp_data_u2)
        # Pressure H1-error
        error_L2_p = form((p1fine_t - p1_fine_interp) ** 2 * dx_test(1) + (p2fine_t - p2_fine_interp) ** 2 * dx_test(2))
        error_L2_p = np.sqrt(assemble_scalar(error_L2_p))
        max_err_L2_p = max(max_err_L2_p, error_L2_p)
        error_grad_p_form = form(ufl.inner(ufl.grad(p1fine_t - p1_fine_interp), ufl.grad(p1fine_t - p1_fine_interp)) * dx_test(1) + ufl.inner(ufl.grad(p2fine_t - p2_fine_interp), ufl.grad(p2fine_t - p2_fine_interp)) * dx_test(2))          
        error_grad_p += assemble_scalar(error_grad_p_form)

        # Fracture pressure H1-error
        error_L2_pc = form((pcfine_t("+") - pc_fine_interp("+")) ** 2 * dS_frac_away(1))
        error_L2_pc = np.sqrt(assemble_scalar(error_L2_pc))
        max_err_L2_pc = max(max_err_L2_pc, error_L2_pc)
        tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
        error_grad_pc_form = form(ufl.inner(ufl.dot(tang, ufl.grad(pcfine_t("+") - pc_fine_interp("+"))), ufl.dot(tang, ufl.grad(pcfine_t("+") - pc_fine_interp("+")))) * dS_frac_away(1))
        error_grad_pc += assemble_scalar(error_grad_pc_form)

        # Displacement H1-error
        error_L2_u = form(ufl.dot(u1fine_t - u1_fine_interp, u1fine_t - u1_fine_interp) * dx_test(1) + ufl.dot(u2fine_t - u2_fine_interp, u2fine_t - u2_fine_interp) * dx_test(2))
        error_L2_u_squared = assemble_scalar(error_L2_u)
        error_grad_u = form(ufl.inner(ufl.grad(u1fine_t - u1_fine_interp), ufl.grad(u1fine_t - u1_fine_interp)) * dx_test(1) + ufl.inner(ufl.grad(u2fine_t - u2_fine_interp), ufl.grad(u2fine_t - u2_fine_interp)) * dx_test(2))
        error_grad_u_squared = assemble_scalar(error_grad_u)
        error_H1_u = np.sqrt(error_L2_u_squared + error_grad_u_squared)
        max_err_H1_u = max(max_err_H1_u, error_H1_u)

    EH1p[i] = max_err_L2_p + np.sqrt(final_time / num_steps * error_grad_p)
    EL2p[i] = max_err_L2_p
    EH1pc[i] = max_err_L2_pc + np.sqrt(final_time / num_steps * error_grad_pc)
    EH1u[i] = max_err_H1_u
print(f"H1-errors displacement: {EH1u}")
print(f"H1-errors pressure: {EH1p}")
print(f"H1-errors fracture pressure: {EH1pc}")
rates_H1_u = np.log(EH1u[1:] / EH1u[:-1]) / np.log(hs[1:] / hs[:-1])
rates_H1_p = np.log(EH1p[1:] / EH1p[:-1]) / np.log(hs[1:] / hs[:-1])
rates_L2_p = np.log(EL2p[1:] / EL2p[:-1]) / np.log(hs[1:] / hs[:-1])
rates_H1_pc = np.log(EH1pc[1:] / EH1pc[:-1]) / np.log(hs[1:] / hs[:-1])
print(f"H1-rates displacement: {rates_H1_u}")
print(f"H1-rates pressure: {rates_H1_p}")
print(f"H1-rates fracture pressure: {rates_H1_pc}")

# Plot errors.
hs_inv = 1 / hs
# print(hs_inv)
# hs_inv = np.array([5.34845195, 10.69690391, 21.39380782, 42.78761563, 85.57523126])
# EH1u = np.array([0.0013034, 0.00089071, 0.00062081, 0.00042625, 0.00027626])
# EH1p = np.array([0.00249461, 0.00164215, 0.00109521, 0.00072685, 0.00046601])
# EH1pc = np.array([0.02570497, 0.02275645, 0.01976688, 0.01656914, 0.01295177])
# Continuous pressure across, fracture
# EH1u = np.array([0.00242556, 0.00136228, 0.00095106, 0.0006546, 0.00042539])
# EH1p = np.array([9.45816752e-04, 4.77325609e-04, 2.39104975e-04, 1.18075638e-04,
#  5.65138430e-05])
# EH1pc = np.array([6.03165119e-04, 3.02332157e-04, 1.49548078e-04, 7.28928632e-05,
#  3.37000541e-05])
# fig, ax = plt.subplots()
# ax.loglog(hs_inv, EH1u, marker='o', color='red')
# ax.loglog(hs_inv, EH1p, marker='o', color='blue')
# ax.loglog(hs_inv, EH1pc, marker='o', color='black')
# ax.legend(["Displacement", "Pressure", "Fracture pressure"], fontsize=12, handlelength=3)
# ax.set_xlabel("Inverse of mesh size (1/h)", fontsize=14)
# ax.grid(which='both', alpha=0.3)
# ax.set_ylabel(r"$H^1$-error", fontsize=14, labelpad=-3)
# ax.set_xticks([5, 1e1, 2e1, 4e1, 8e1])
# ax.set_ylim([2e-5, 3e-3])
# ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax.tick_params(labelsize=14.5)
# draw_multiple_loglog_slopes(fig, ax, origin=(0.85 * hs_inv[-1], 0.95 * EH1u[-1]), triangle_width=1, inverted=True, slopes=[-0.62], labelcolor=(0.33, 0.33, 0.33))
# draw_multiple_loglog_slopes(fig, ax, origin=(0.45 * hs_inv[-1], 1.3 * EH1p[-2]), triangle_width=1, inverted=False, slopes=[-1.06], labelcolor=(0.33, 0.33, 0.33))
# draw_multiple_loglog_slopes(fig, ax, origin=(0.85 * hs_inv[-1], 0.95 * EH1pc[-1]), triangle_width=1, inverted=True, slopes=[-1.11], labelcolor=(0.33, 0.33, 0.33))   
# plt.savefig("H1_errors_space_cont_pressure.png")