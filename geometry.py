import dolfinx.fem
import dolfinx.fem.petsc
from dolfinx import plot, fem, default_real_type, default_scalar_type
import dolfinx.io
import gmsh
import mpi4py.MPI
import numpy as np
import petsc4py.PETSc
import ufl
import pyvista as pv
pv.off_screen = True
from viskex import DolfinxPlotter
from basix.ufl import element, mixed_element

import multiphenicsx.fem
import multiphenicsx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem

r = 3
mesh_size = 1. / 16.

gmsh.initialize()
gmsh.model.add("mesh")

# Create rectangle
pll = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
plr = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, mesh_size)
pul = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, mesh_size)
pur = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, mesh_size)
l1 = gmsh.model.geo.addLine(pll, plr)
l2 = gmsh.model.geo.addLine(plr, pur)
l3 = gmsh.model.geo.addLine(pur, pul)
l4 = gmsh.model.geo.addLine(pul, pll)
square_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

# Fracture
fracx_left = 0.3
fracx_right = 0.7
fracy = 0.5
pfr1 = gmsh.model.geo.addPoint(fracx_left, fracy, 0.0, mesh_size)
pfr2 = gmsh.model.geo.addPoint(fracx_right, fracy, 0.0, mesh_size)
midpoint_left = gmsh.model.geo.addPoint(0.0, fracy, 0.0, mesh_size)
midpoint_right = gmsh.model.geo.addPoint(1.0, fracy, 0.0, mesh_size)
frac = gmsh.model.geo.addLine(pfr1, pfr2)
upper_boundary_to_midpoint_left = gmsh.model.geo.addLine(pul, midpoint_left)
midpoint_right_to_upper_boundary = gmsh.model.geo.addLine(midpoint_right, pur)
midpoint_left_to_lower_boundary = gmsh.model.geo.addLine(midpoint_left, pll)
lower_boundary_to_midpoint_right = gmsh.model.geo.addLine(plr, midpoint_right)
mid_to_fracture_left = gmsh.model.geo.addLine(midpoint_left, pfr1)
fracture_to_mid_right = gmsh.model.geo.addLine(pfr2, midpoint_right)

# Matrix subdomains, split into a positive and negative side relative to the fracture
upper_boundary = gmsh.model.geo.addCurveLoop([upper_boundary_to_midpoint_left, mid_to_fracture_left, frac, fracture_to_mid_right, midpoint_right_to_upper_boundary, l3])
upper_rectangle = gmsh.model.geo.addPlaneSurface([upper_boundary])
lower_boundary = gmsh.model.geo.addCurveLoop([midpoint_left_to_lower_boundary, l1, lower_boundary_to_midpoint_right, -fracture_to_mid_right, -frac, -mid_to_fracture_left])
lower_rectangle = gmsh.model.geo.addPlaneSurface([lower_boundary])

gmsh.model.geo.synchronize()
# Matrix subdomains
gmsh.model.addPhysicalGroup(2, [upper_rectangle], 1)
gmsh.model.addPhysicalGroup(2, [lower_rectangle], 2)
# Left boundary
gmsh.model.addPhysicalGroup(1, [upper_boundary_to_midpoint_left, midpoint_left_to_lower_boundary], 1)
# Right boundary
gmsh.model.addPhysicalGroup(1, [lower_boundary_to_midpoint_right, midpoint_right_to_upper_boundary], 2)
# Bottom boundary
gmsh.model.addPhysicalGroup(1, [l1], 3) 
# Top boundary
gmsh.model.addPhysicalGroup(1, [l3], 4)
# Fracture and interfaces between matrix subdomains
gmsh.model.addPhysicalGroup(1, [mid_to_fracture_left, fracture_to_mid_right], 5)
gmsh.model.addPhysicalGroup(1, [frac], 6)
gmsh.model.mesh.generate(2)

partitioner = dolfinx.mesh.create_cell_partitioner(  # type: ignore[call-arg, unused-ignore]
    dolfinx.mesh.GhostMode.shared_facet, 2)
mesh, subdomains, boundaries_and_interfaces, *_ = dolfinx.io.gmsh.model_to_mesh(
    gmsh.model, comm=mpi4py.MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner)
gmsh.finalize()
assert subdomains is not None
assert boundaries_and_interfaces is not None

# Create connectivities required by the rest of the code
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

cells_omega = subdomains.indices
cells_omega_plus = subdomains.indices[subdomains.values == 1]
cells_omega_minus = subdomains.indices[subdomains.values == 2]
facets_left = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 1]
facets_right = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 2]
facets_bottom = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 3]
facets_top = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 4]
facets_interface_not_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 5]
facets_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]

# Define associated measures
ds = ufl.Measure("ds", subdomain_data=boundaries_and_interfaces)

# Define a custom measure on the fracture facets to ensure a consistent ordering between + and - sides.
integration_entities_on_gamma = dolfinx.fem.compute_integration_domains(
    dolfinx.fem.IntegralType.interior_facet, mesh.topology, facets_frac)
integration_entities_on_gamma_reshaped = integration_entities_on_gamma.reshape(-1, 4)
connected_cells_to_gamma = integration_entities_on_gamma_reshaped[:, [0, 2]]
subdomain_ordering = (
    subdomains.values[connected_cells_to_gamma[:, 0]] < subdomains.values[connected_cells_to_gamma[:, 1]])
if len(subdomain_ordering) > 0 and any(subdomain_ordering):
    integration_entities_on_gamma_reshaped[subdomain_ordering] = integration_entities_on_gamma_reshaped[
        subdomain_ordering][:, [2, 3, 0, 1]]
integration_entities_on_gamma = integration_entities_on_gamma_reshaped.flatten()
dS = ufl.Measure("dS", domain=mesh, subdomain_data=[(2, np.array(integration_entities_on_gamma, dtype=np.int32))])
dS = dS(2)
n = ufl.FacetNormal(mesh)


# Function spaces
P1_displacement = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type)
P1_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
P1_frac_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
W = fem.functionspace(mesh, mixed_element([P1_displacement, P1_pressure, P1_frac_pressure]))
V0 = W.sub(0)
Q0 = W.sub(1)
Qc0 = W.sub(2)
V, map_V = V0.collapse()
Q, map_Q = Q0.collapse()
Qc, map_Qc = Qc0.collapse()
w = fem.Function(W)  # Current solution     
w_n = fem.Function(W)  # Solution at previous timestep
(u, p, pc) = ufl.split(w)  # Split into displacement and pressure
(u_n, p_n, pc_n) = ufl.split(w_n)
(v, q, qc) = ufl.TestFunctions(W)  # Test functions for displacement and pressure

# Define restrictions
fdim = boundaries_and_interfaces.dim
dofs_frac = fem.locate_dofs_topological(Qc, fdim, facets_frac)
dofs_Q_omega = fem.locate_dofs_topological(Q, subdomains.dim, cells_omega)
dofs_V_omega = fem.locate_dofs_topological(V, subdomains.dim, cells_omega)
dofs_frac = map_Qc[dofs_frac]
dofs_Q_omega = map_Q[dofs_Q_omega]
dofs_V_omega = map_V[dofs_V_omega]
dofs_restricted = np.unique(
    np.concatenate([
        dofs_V_omega,
        dofs_Q_omega,
        dofs_frac
    ])
).astype(np.int32)
restriction_mixed = multiphenicsx.fem.DofMapRestriction(W.dofmap, dofs_restricted)
restriction = [restriction_mixed]

# Define boundary conditions
zero = fem.Constant(mesh, default_scalar_type(0))
one = fem.Constant(mesh, default_scalar_type(1))
zero_vector = fem.Constant(mesh, default_scalar_type((0,) * mesh.geometry.dim))
u_D = fem.Function(V)
p_D = fem.Function(Q)
p_D2 = fem.Function(Q)
u_D.x.array[:] = 0
p_D.x.array[:] = 0
p_D2.x.array[:] = 1
dofs_V = fem.locate_dofs_topological((V0, V), fdim, np.concatenate([facets_top, facets_bottom]))
dofs_Q = fem.locate_dofs_topological((Q0, Q), fdim, facets_bottom)
dofs_Q2 = fem.locate_dofs_topological((Q0, Q), fdim, facets_top)
bc_u = fem.dirichletbc(u_D, dofs_V, V0)
bc_p = fem.dirichletbc(p_D, dofs_Q, Q0)
bc_p2 = fem.dirichletbc(p_D2, dofs_Q2, Q0)
bc = [bc_u, bc_p, bc_p2]

# Material parameters
lame_lambda = 1.0  # Lamé's first parameter
G = 1.0  # Shear modulus (Lamé's second parameter)
gamma = 1.0  # Viscoelastic parameter
alpha = 1.0  # Biot coefficient
c_0 = 1.0  # Specific storage coefficient
c_f = 1.0  # Fluid compressibility
mu = 1.0  # Fluid viscosity
k = 1.0  # Permeability
mu_fr = 1.0  # Friction coefficient

# Time stepping parameters
t = 0.0
T = 1.0
num_steps = 10
dt = T / num_steps

# Useful funcions
eps = fem.Constant(mesh, default_scalar_type(1e-10))
def epsilon(u):
    return ufl.sym(
        ufl.grad(u)
    )  # Strain tensor

def ppos(x):   # Positive part of x, used for contact condition.
    return (x + ufl.sqrt(x**2 + eps)) / 2.

def maximum(x, y):
    return 0.5 * (x + y + ufl.sqrt((x - y)**2))

def g(x):  # Initial fracture gap.
    return 1e-4 * x[0]

def indicator_function(x):
    # Indicator function, used to distinguish between open and closed cells.
    # Equals 1 if x>0, 0 if x<0 and 1/2 if x=0.
    return 0.5 * (ufl.sign(x) + 1) 

# Contact condition
x = ufl.SpatialCoordinate(mesh)
pen = 1e4  # Penalty/stiffness parameter
u_jump_norm = ufl.inner(n("+"), u("+") - u("-"))  # Normal jump of displacement across the fracture
u_jump_tang = u("+") - u("-") - u_jump_norm * n("+")  # Tangential jump of displacement across the fracture
v_jump_norm = ufl.inner(n("+"), v("+") - v("-"))
v_jump_tang = v("+") - v("-") - v_jump_norm * n("+")
u_jump_tang_prev = u_n("+") - u_n("-") - ufl.inner(n("+"), u_n("+") - u_n("-")) * n("+")
norm_traction = - pen * ppos(u_jump_norm - g(x))  # Normal compliance law
trial = pen * (u_jump_tang - u_jump_tang_prev)
norm_trial = ufl.sqrt(ufl.inner(trial, trial) + eps)
friction_bound = -mu_fr * norm_traction
chi = indicator_function(-friction_bound)
one = fem.Constant(mesh, default_scalar_type(1))
tang_traction = ((chi - one) * friction_bound * trial) / ((chi - one) * maximum(friction_bound, norm_trial) + chi)
friction = True
if friction:
    contact_condition = norm_traction * v_jump_norm * dS + ufl.inner(tang_traction, v_jump_tang) * dS   # Frictional contact condition
else:
    contact_condition = norm_traction * v_jump_norm * dS  # Frictionless contact condition


# Mechanics equation
f = fem.Constant(mesh, default_scalar_type((0.0, 0.0)))  # Source term (body force)
a = 2 * G * ufl.inner(epsilon(u), epsilon(v)) * ufl.dx + lame_lambda * ufl.inner(ufl.div(u), ufl.div(v)) * ufl.dx
b = gamma * ufl.inner(epsilon((u - u_n) / dt), epsilon(v)) * ufl.dx + gamma * ufl.inner(ufl.div((u - u_n) / dt), ufl.div(v)) * ufl.dx
coupling_term = -alpha * ufl.inner(p, ufl.div(v)) * ufl.dx
coupling_term_fracture = ufl.inner(pc("+"), v_jump_norm) * dS
source_mechanics = ufl.inner(f, v) * ufl.dx
mechanics_equation = a + b + coupling_term + coupling_term_fracture - source_mechanics

# Flow equation in matrix
accumulation = ufl.inner(c_0 * ((p - p_n) / dt) + alpha * ufl.div((u - u_n) / dt), q) * ufl.dx
flux = k / mu * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
f_fluid = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow = ufl.inner(f_fluid, q) * ufl.dx
matrix_fracture_coupling = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p("+")),q("+")) * dS + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p("-")),q("-")) * dS
fluid_equation = accumulation + flux - source_flow - matrix_fracture_coupling

# Flow equation in fracture
tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
accumulation_fracture = ufl.inner(c_f * g(x) * ((pc("+") - pc_n("+")) / dt) - ufl.inner(((u("+") - u_n("+") - u("-") + u_n("-")) / dt), n("+")), qc("+")) * dS
flux_fracture = 1 / (12 * mu) * ufl.inner(g(x)**3 * ufl.dot(ufl.grad(pc("+")), tang), ufl.dot(ufl.grad(qc("+")), tang)) * dS
f_fluid_fracture = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow_fracture = ufl.inner(g(x) * f_fluid_fracture, qc("+")) * dS
matrix_fracture_coupling_2 = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p("+")), qc("+")) * dS + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p("-")), qc("+")) * dS
fracture_fluid_equation = accumulation_fracture + flux_fracture - source_flow_fracture + matrix_fracture_coupling_2

# Full equation system
F = mechanics_equation + fluid_equation + fracture_fluid_equation
# F = [mechanics_equation, fluid_equation, fracture_fluid_equation]

# Initial condition
def initial_condition_pressure(x):
    return 0 * x[0]

def initial_condition_displacement(x):
    return (0 * x[0], 0 * x[0])
w_n.sub(0).interpolate(initial_condition_displacement)
w_n.sub(1).interpolate(initial_condition_pressure)
w_n.sub(2).interpolate(initial_condition_pressure)
# Solver setup
petsc_options = {
"snes_type": "newtonls",
"snes_linesearch_type": "none",
"snes_max_it": 100,
"snes_atol": 1e-6,
"snes_rtol": 1e-6,
"ksp_error_if_not_converged": True,
"ksp_type": "preonly",
"pc_type": "lu",
}

problem = multiphenicsx.fem.petsc.NonlinearProblem(
    [F], [w], bcs=bc,
    petsc_options_prefix="mixed_dimensional_", petsc_options=petsc_options,
    kind="mpi", restriction=restriction
)

# Time-stepping loop
step = 0
utimes = []
ptimes = []
pctimes = []
utimes.append(w.sub(0).collapse())
ptimes.append(w.sub(1).collapse())
pctimes.append(w.sub(2).collapse())
for _ in range(num_steps):
    t += dt
    _ = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    print(converged_reason)
    assert converged_reason > 0
    w_n.x.array[:] = w.x.array
    utimes.append(w.sub(0).collapse())
    ptimes.append(w.sub(1).collapse())
    pctimes.append(w.sub(2).collapse())
    step += 1

# # Plotting
uplot = utimes[-1]
pplot = ptimes[-1]
pcplot = pctimes[-1]
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(uplot.function_space)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# plotter = pv.Plotter()
p_topology, p_cell_types, p_geometry = plot.vtk_mesh(pplot.function_space)
p_grid = pv.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
plotter = pv.Plotter()
p_grid.point_data["p"] = pplot.x.array
plotter.add_mesh(
    p_grid
)
plotter.view_xy()
plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
plotter.show(screenshot=f"mixed_biot_matrix.png")

# # # Map pressure dofs to the same ordering as displacement dofs, so that they can be plotted together. 
# # u_coords = uplot.function_space.tabulate_dof_coordinates()
# # p_coords = pplot.function_space.tabulate_dof_coordinates()
# # u_idx = np.lexsort(u_coords.T)
# # p_idx = np.lexsort(p_coords.T)
# # mapped_p = np.zeros_like(pplot.x.array)
# # mapped_p[u_idx] = pplot.x.array[p_idx]

# # # u_grid.point_data["u"] = uplot.x.array.reshape((-1,2))
# # # plotter.add_mesh(
# # #     u_grid,
# # #     scalars="u",
# # #     component=0,            
# # #     show_edges=False,
# # # )
# # # plotter.view_xy()
# # # plotter.show(screenshot=f"mixed_biot_matrix.png")

# # # Pad displacement vector with zero z-component
# # u = uplot.x.array.reshape((-1, mesh.geometry.dim)).copy()
# # u_3d = np.zeros((u.shape[0], 3))
# # u_3d[:, :2] = u
# # u_grid.point_data["u"] = u_3d
# # warped = u_grid.warp_by_vector("u", factor=1)
# # warped.point_data["p"] = mapped_p
# # plotter.add_mesh(
# #     warped,
# #     scalars="p",            
# #     show_edges=False,
# # )
# # plotter.view_xy()
# # plotter.show(screenshot=f"mixed_biot_matrix.png")

plotter = pv.Plotter()
pc_topology, pc_cell_types, pc_geometry = plot.vtk_mesh(pcplot.function_space)
pc_grid = pv.UnstructuredGrid(pc_topology, pc_cell_types, pc_geometry)
pc_grid.point_data["pc"] = pcplot.x.array
plotter.add_mesh(
    pc_grid, show_edges=True
)
plotter.view_xy()
plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
plotter.show(screenshot=f"mixed_biot_fracture.png")