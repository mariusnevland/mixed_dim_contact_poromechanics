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

cells_Omega = subdomains.indices
cells_Omega1 = subdomains.indices[subdomains.values == 1]
cells_Omega2 = subdomains.indices[subdomains.values == 2]
facets_left = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 1]
facets_right = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 2]
facets_bottom = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 3]
facets_top = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 4]
facets_interface_not_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 5]
facets_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]

# Define associated measures
dx = ufl.Measure("dx", subdomain_data=subdomains)

def ordered_interior_entities(facets):

    entities = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet,
        mesh.topology,
        facets
    )

    entities = entities.reshape(-1, 4)

    if len(entities) == 0:
        return np.array([], dtype=np.int32)

    connected_cells = entities[:, [0, 2]]

    # We want:
    # Omega1 (tag 1) = "+"
    # Omega2 (tag 2) = "-"
    swap = (
        subdomains.values[connected_cells[:, 0]]
        >
        subdomains.values[connected_cells[:, 1]]
    )

    entities[swap] = entities[swap][:, [2, 3, 0, 1]]

    return entities.flatten().astype(np.int32)

entities_interface = ordered_interior_entities(
    facets_interface_not_frac
)

entities_fracture = ordered_interior_entities(
    facets_frac
)
dS_data = [
    (5, entities_interface),
    (6, entities_fracture),
]

dS_all = ufl.Measure(
    "dS",
    domain=mesh,
    subdomain_data=dS_data
)
dS_intf = dS_all(5)
dS_frac = dS_all(6)

# Mixed function spaces
P1_displacement = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type)
P1_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
P1_frac_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
W = fem.functionspace(mesh, mixed_element([P1_displacement, P1_displacement, P1_displacement,
                                           P1_pressure, P1_pressure, P1_pressure,
                                           P1_frac_pressure]))
V1_0 = W.sub(0)  # Displacement on Omega1
V2_0 = W.sub(1)  # Displacement on Omega2
MV_0 = W.sub(2)  # Lagrange multiplier space for displacement
Q1_0 = W.sub(3)  # Pressure on Omega1
Q2_0 = W.sub(4)  # Pressure on Omega2
MQ_0 = W.sub(5)  # Lagrange multiplier space for pressure
Qc0 = W.sub(6)  # Fracture pressure
V1, map_V1 = V1_0.collapse()
V2, map_V2 = V2_0.collapse()
MV, map_MV = MV_0.collapse()
Q1, map_Q1 = Q1_0.collapse()
Q2, map_Q2 = Q2_0.collapse()
MQ, map_MQ = MQ_0.collapse()
Qc, map_Qc = Qc0.collapse()
w = fem.Function(W)  # Current solution     
w_n = fem.Function(W)  # Solution at previous timestep
(u1, u2, lu, p1, p2, lp, pc) = ufl.split(w)
(u1_n, u2_n, lu_n, p1_n, p2_n, lp_n, pc_n) = ufl.split(w_n)
(v1, v2, mv, q1, q2, mq, qc) = ufl.TestFunctions(W)


# Define restrictions
fdim = boundaries_and_interfaces.dim
# Locate vector DOF blocks on Omega1
blocks_V1_Omega1 = fem.locate_dofs_topological(
    V1,
    subdomains.dim,
    cells_Omega1
)
# Locate vector DOF blocks on Omega2
blocks_V2_Omega2 = fem.locate_dofs_topological(
    V2,
    subdomains.dim,
    cells_Omega2
)
blocks_M_Gamma = fem.locate_dofs_topological(
    MV,
    fdim,
    facets_interface_not_frac
)
bs1 = V1.dofmap.index_map_bs
bs2 = V2.dofmap.index_map_bs
bs3 = MV.dofmap.index_map_bs
# Expand block indices -> scalar/unrolled DOF indices
scalar_V1_Omega1 = (
    bs1 * blocks_V1_Omega1[:, None]
    + np.arange(bs1)
).reshape(-1)

scalar_V2_Omega2 = (
    bs2 * blocks_V2_Omega2[:, None]
    + np.arange(bs2)
).reshape(-1)

scalar_MV_Gamma = (
    bs3 * blocks_M_Gamma[:, None]
    + np.arange(bs3)
).reshape(-1)
# Now map collapsed scalar DOFs -> mixed-space scalar DOFs
dofs_V1_Omega1 = map_V1[scalar_V1_Omega1]
dofs_V2_Omega2 = map_V2[scalar_V2_Omega2]
dofs_MV_Gamma = map_MV[scalar_MV_Gamma]
dofs_Q1_Omega1 = map_Q1[fem.locate_dofs_topological(Q1, subdomains.dim, cells_Omega1)]
dofs_Q2_Omega2 = map_Q2[fem.locate_dofs_topological(Q2, subdomains.dim, cells_Omega2)]
dofs_MQ_Gamma = map_MQ[fem.locate_dofs_topological(MQ, fdim, facets_interface_not_frac)]
dofs_Qc_frac = map_Qc[fem.locate_dofs_topological(Qc, fdim, facets_frac)]
dofs_restricted = np.unique(
    np.concatenate([
        dofs_V1_Omega1,
        dofs_V2_Omega2,
        dofs_MV_Gamma,
        dofs_Q1_Omega1,
        dofs_Q2_Omega2,
        dofs_MQ_Gamma,
        dofs_Qc_frac
    ])
).astype(np.int32)
restriction_mixed = multiphenicsx.fem.DofMapRestriction(W.dofmap, dofs_restricted)
restriction = [restriction_mixed]

# Boundary conditions
u_D1 = fem.Function(V1)
u_D2 = fem.Function(V2)
p_D1 = fem.Function(Q1)
p_D2 = fem.Function(Q2)
u_D1.x.array[:] = 1
u_D2.x.array[:] = 0
p_D1.x.array[:] = 1
p_D2.x.array[:] = 0
dofs_V1 = fem.locate_dofs_topological((V1_0, V1), fdim, facets_top)
dofs_V2 = fem.locate_dofs_topological((V2_0, V2), fdim, facets_bottom)
dofs_Q1 = fem.locate_dofs_topological((Q1_0, Q1), fdim, facets_top)
dofs_Q2 = fem.locate_dofs_topological((Q2_0, Q2), fdim, facets_bottom)
bc_u1 = fem.dirichletbc(u_D1, dofs_V1, V1_0)
bc_u2 = fem.dirichletbc(u_D2, dofs_V2, V2_0)
bc_p1 = fem.dirichletbc(p_D1, dofs_Q1, Q1_0)
bc_p2 = fem.dirichletbc(p_D2, dofs_Q2, Q2_0)
bc = [bc_u1, bc_u2, bc_p1, bc_p2]

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

x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
# Contact condition
pen = 1e6  # Penalty/stiffness parameter
u_jump_norm = ufl.inner(n("+"), u1("+") - u2("-"))  # Normal jump of displacement across the fracture
u_jump_tang = u1("+") - u2("-") - u_jump_norm * n("+")  # Tangential jump of displacement across the fracture
v_jump_norm = ufl.inner(n("+"), v1("+") - v2("-"))
v_jump_tang = v1("+") - v2("-") - v_jump_norm * n("+")
u_jump_tang_prev = u1_n("+") - u2_n("-") - ufl.inner(n("+"), u1_n("+") - u2_n("-")) * n("+")
norm_traction = - pen * ppos(u_jump_norm - g(x))  # Normal compliance law
trial = pen * (u_jump_tang - u_jump_tang_prev)
norm_trial = ufl.sqrt(ufl.inner(trial, trial) + eps)
friction_bound = -mu_fr * norm_traction
chi = indicator_function(-friction_bound)
one = fem.Constant(mesh, default_scalar_type(1))
tang_traction = ((chi - one) * friction_bound * trial) / ((chi - one) * maximum(friction_bound, norm_trial) + chi)
friction = True
if friction:
    contact_condition = norm_traction * v_jump_norm * dS_frac + ufl.inner(tang_traction, v_jump_tang) * dS_frac   # Frictional contact condition
else:
    contact_condition = norm_traction * v_jump_norm * dS_frac  # Frictionless contact condition

# Mechanics equation, the equations for Omega1 and Omega2 are added.
f = fem.Constant(mesh, default_scalar_type((0.0, 0.0)))  # Source term (body force)
A1 = 2 * G * ufl.inner(epsilon(u1), epsilon(v1)) * dx(1) + lame_lambda * ufl.inner(ufl.div(u1), ufl.div(v1)) * dx(1)
A2 = 2 * G * ufl.inner(epsilon(u2), epsilon(v2)) * dx(2) + lame_lambda * ufl.inner(ufl.div(u2), ufl.div(v2)) * dx(2)
B1 = gamma * ufl.inner(epsilon((u1 - u1_n) / dt), epsilon(v1)) * dx(1) + gamma * ufl.inner(ufl.div((u1 - u1_n) / dt), ufl.div(v1)) * dx(1)
B2 = gamma * ufl.inner(epsilon((u2 - u2_n) / dt), epsilon(v2)) * dx(2) + gamma * ufl.inner(ufl.div((u2 - u2_n) / dt), ufl.div(v2)) * dx(2)
coupling_term1 = -alpha * ufl.inner(p1, ufl.div(v1)) * dx(1)
coupling_term2 = -alpha * ufl.inner(p2, ufl.div(v2)) * dx(2)
coupling_term_fracture = ufl.inner(pc("+"), v_jump_norm) * dS_frac
source_mechanics1 = ufl.inner(f, v1) * dx(1)
source_mechanics2 = ufl.inner(f, v2) * dx(2)
interface_term = ufl.inner(lu("+"), v1("+") - v2("-")) * dS_intf + ufl.inner(mv("+"), u1("+") - u2("-")) * dS_intf
mechanics_equation = A1 + A2 + B1 + B2 + coupling_term1 + coupling_term2 + coupling_term_fracture - source_mechanics1 - source_mechanics2 - interface_term - contact_condition

# Mass balance in the matrix, the equations for Omega1 and Omega2 are added.
accumulation1 = ufl.inner(c_0 * ((p1 - p1_n) / dt) + alpha * ufl.div((u1 - u1_n) / dt), q1) * dx(1)
accumulation2 = ufl.inner(c_0 * ((p2 - p2_n) / dt) + alpha * ufl.div((u2 - u2_n) / dt), q2) * dx(2)
flux1 = k / mu * ufl.inner(ufl.grad(p1), ufl.grad(q1)) * dx(1)
flux2 = k / mu * ufl.inner(ufl.grad(p2), ufl.grad(q2)) * dx(2)
f_fluid = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow1 = ufl.inner(f_fluid, q1) * dx(1)
source_flow2 = ufl.inner(f_fluid, q2) * dx(2)
interface_term = ufl.inner(lp("+"), q1("+") - q2("-")) * dS_intf + ufl.inner(mq("+"), p1("+") - p2("-")) * dS_intf
matrix_fracture_coupling = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p1("+")),q1("+")) * dS_frac + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p2("-")),q2("-")) * dS_frac
fluid_equation = accumulation1 + accumulation2 + flux1 + flux2 - source_flow1 - source_flow2 + interface_term - matrix_fracture_coupling

# Mass balance in the fracture
tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
accumulation_fracture = ufl.inner(c_f * g(x) * ((pc("+") - pc_n("+")) / dt) - ufl.inner(((u1("+") - u1_n("+") - u2("-") + u2_n("-")) / dt), n("+")), qc("+")) * dS_frac
flux_fracture = 1 / (12 * mu) * ufl.inner(g(x)**3 * ufl.dot(ufl.grad(pc("+")), tang), ufl.dot(ufl.grad(qc("+")), tang)) * dS_frac
f_fluid_fracture = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow_fracture = ufl.inner(g(x) * f_fluid_fracture, qc("+")) * dS_frac
matrix_fracture_coupling_2 = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p1("+")), qc("+")) * dS_frac + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p2("-")), qc("+")) * dS_frac
fracture_fluid_equation = accumulation_fracture + flux_fracture - source_flow_fracture + matrix_fracture_coupling_2

# Full equation system
F = mechanics_equation + fluid_equation + fracture_fluid_equation

# Initial condition
def initial_condition_pressure(x):
    return 0 * x[0]

def initial_condition_displacement(x):
    return (0 * x[0], 0 * x[0])
w_n.sub(0).interpolate(initial_condition_displacement)
w_n.sub(1).interpolate(initial_condition_displacement)
w_n.sub(2).interpolate(initial_condition_displacement)
w_n.sub(3).interpolate(initial_condition_pressure)
w_n.sub(4).interpolate(initial_condition_pressure)
w_n.sub(5).interpolate(initial_condition_pressure)
w_n.sub(6).interpolate(initial_condition_pressure)

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
"pc_factor_mat_solver_type": "mumps",
"mat_mumps_icntl_7": 4
}

problem = multiphenicsx.fem.petsc.NonlinearProblem(
    [F], [w], bcs=bc,
    petsc_options_prefix="mixed_dimensional_", petsc_options=petsc_options,
    kind="mpi", restriction=restriction
)

# Time-stepping loop
step = 0
u1times = []
u2times = []
lutimes = []
p1times = []
p2times = []
lptimes = []
pctimes = []
u1times.append(w.sub(0).collapse())
u2times.append(w.sub(1).collapse())
lutimes.append(w.sub(2).collapse())
p1times.append(w.sub(3).collapse())
p2times.append(w.sub(4).collapse())
lptimes.append(w.sub(5).collapse())
pctimes.append(w.sub(6).collapse())
for _ in range(num_steps):
    t += dt
    _ = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    assert converged_reason > 0
    num_iterations = problem.solver.getIterationNumber()
    print(f"Time step {step}: {converged_reason=} {num_iterations=}")
    w_n.x.array[:] = w.x.array  # Update solution at previous time step
    u1times.append(w.sub(0).collapse())
    u2times.append(w.sub(1).collapse())
    lutimes.append(w.sub(2).collapse())
    p1times.append(w.sub(3).collapse())
    p2times.append(w.sub(4).collapse())
    lptimes.append(w.sub(5).collapse())
    pctimes.append(w.sub(6).collapse())
    step += 1


# # Plotting
u1plot = u1times[-1]
u2plot = u2times[-1]
pplot1 = p1times[-1]
pplot2 = p2times[-1]
pcplot = pctimes[-1]
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(uplot.function_space)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# plotter = pv.Plotter()
p_topology, p_cell_types, p_geometry = plot.vtk_mesh(pplot1.function_space)
p_grid = pv.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
plotter = pv.Plotter()
p_grid.point_data["p"] = pplot1.x.array
plotter.add_mesh(
    p_grid, show_edges=True
)
plotter.view_xy()
plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
plotter.show(screenshot=f"mixed_biot_matrix_upper.png")

plotter = pv.Plotter()
p_topology, p_cell_types, p_geometry = plot.vtk_mesh(pplot2.function_space)
p_grid = pv.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
plotter = pv.Plotter()
p_grid.point_data["p"] = pplot2.x.array
plotter.add_mesh(
    p_grid, show_edges=True
)
plotter.view_xy()
plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
plotter.show(screenshot=f"mixed_biot_matrix_lower.png")

plotter = pv.Plotter()
p_topology, p_cell_types, p_geometry = plot.vtk_mesh(pplot2.function_space)
p_grid = pv.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
plotter = pv.Plotter()
p_grid.point_data["p"] = pplot1.x.array + pplot2.x.array
plotter.add_mesh(
    p_grid, show_edges=True
)
plotter.view_xy()
plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
plotter.show(screenshot=f"mixed_biot_matrix_sum.png")

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

# plotter = pv.Plotter()
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u1plot.function_space)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = u1plot.x.array.reshape((-1, 2))
# plotter.add_mesh(
#     u_grid, scalars="u", component=0, show_edges=True
# )
# plotter.view_xy()
# plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
# plotter.show(screenshot=f"mixed_biot_disp_upper.png")

# plotter = pv.Plotter()
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u2plot.function_space)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = u2plot.x.array.reshape((-1, 2))
# plotter.add_mesh(
#     u_grid, scalars="u", component=0, show_edges=True
# )
# plotter.view_xy()
# plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
# plotter.show(screenshot=f"mixed_biot_disp_lower.png")
 
# Pad displacement vector with zero z-component
# plotter = pv.Plotter()
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u2plot.function_space)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u = u2plot.x.array.reshape((-1, mesh.geometry.dim)).copy()
# u_3d = np.zeros((u.shape[0], 3))
# u_3d[:, :2] = u
# u_grid.point_data["u"] = u_3d
# warped = u_grid.warp_by_vector("u", factor=1)
# plotter.add_mesh(
#     warped,           
#     show_edges=False,
# )
# plotter.view_xy()
# plotter.show(screenshot=f"mixed_biot_disp_warped2.png")

