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
V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))  # Space for the displacement
V1 = V.clone()
V2 = V.clone()
V_intf = V.clone()
Q = fem.functionspace(mesh, ("Lagrange", 1))  # Space for the matrix pressure
Q1 = Q.clone()
Q2 = Q.clone()
Q_intf = Q.clone()
Qc = Q.clone()  # Space for the fracture pressure
(u1, p1, v1, q1) = (fem.Function(V1), fem.Function(Q1), ufl.TestFunction(V1), ufl.TestFunction(Q1))
(u2, p2, v2, q2) = (fem.Function(V2), fem.Function(Q2), ufl.TestFunction(V2), ufl.TestFunction(Q2))
(pc, qc) = (fem.Function(Qc), ufl.TestFunction(Qc))
(u1_n, p1_n, u2_n, p2_n, pc_n) = (fem.Function(V1), fem.Function(Q1), fem.Function(V2), fem.Function(Q2), fem.Function(Qc))
(u_intf, p_intf, v_intf, q_intf) = (fem.Function(V_intf), fem.Function(Q_intf), ufl.TestFunction(V_intf), ufl.TestFunction(Q_intf))  # Possibly need some sort of Lagrange multipliers on the interface (not including the fracture) to enforce continuity across these interfaces.

# Define restrictions
fdim = boundaries_and_interfaces.dim
dofs_V1_Omega1 = fem.locate_dofs_topological(V1, subdomains.dim, cells_Omega1)
dofs_V2_Omega2 = fem.locate_dofs_topological(V2, subdomains.dim, cells_Omega2)
dofs_Q1_Omega1 = fem.locate_dofs_topological(Q1, subdomains.dim, cells_Omega1)
dofs_Q2_Omega2 = fem.locate_dofs_topological(Q2, subdomains.dim, cells_Omega2)
dofs_V_intf = fem.locate_dofs_topological(V_intf, fdim, facets_interface_not_frac)
dofs_Q_intf = fem.locate_dofs_topological(Q_intf, fdim, facets_interface_not_frac)
dofs_frac = fem.locate_dofs_topological(Qc, fdim, facets_frac)
restriction_V1_Omega1 = multiphenicsx.fem.DofMapRestriction(V1.dofmap, dofs_V1_Omega1)
restriction_V2_Omega2 = multiphenicsx.fem.DofMapRestriction(V2.dofmap, dofs_V2_Omega2)
restriction_Q1_Omega1 = multiphenicsx.fem.DofMapRestriction(Q1.dofmap, dofs_V1_Omega1)
restriction_Q2_Omega2 = multiphenicsx.fem.DofMapRestriction(Q2.dofmap, dofs_V2_Omega2)
restriction_V_intf = multiphenicsx.fem.DofMapRestriction(V_intf.dofmap, dofs_V_intf)
restriction_Q_intf = multiphenicsx.fem.DofMapRestriction(Q_intf.dofmap, dofs_Q_intf)
restriction_Qc_frac = multiphenicsx.fem.DofMapRestriction(Qc.dofmap, dofs_frac)
restriction = [restriction_V1_Omega1, restriction_V2_Omega2, restriction_Q1_Omega1, restriction_Q2_Omega2, restriction_Qc_frac]

# Boundary conditions
zero = dolfinx.fem.Constant(mesh, default_scalar_type(0))
one = dolfinx.fem.Constant(mesh, default_scalar_type(1))
zero_vector = dolfinx.fem.Constant(mesh, default_scalar_type((0,) * mesh.geometry.dim))
u_D = fem.Function(V)
p_D = fem.Function(Q)
u_D.x.array[:] = 0
p_D.x.array[:] = 0
dofs_V1 = fem.locate_dofs_topological(V1, fdim, facets_top)
dofs_V2 = fem.locate_dofs_topological(V2, fdim, facets_bottom)
dofs_Q1 = fem.locate_dofs_topological(Q1, fdim, facets_top)
dofs_Q2 = fem.locate_dofs_topological(Q2, fdim, facets_bottom)
bc_u1 = fem.dirichletbc(zero_vector, dofs_V1, V1)
bc_u2 = fem.dirichletbc(zero_vector, dofs_V2, V2)
bc_p1 = fem.dirichletbc(one, dofs_Q1, Q1)
bc_p2 = fem.dirichletbc(zero, dofs_Q2, Q2)
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
f = fem.Constant(mesh, default_scalar_type((0.0, 0.0)))  # Source term (body force)
v_jump_norm = ufl.inner(n("+"), v1("+") - v2("-"))
# Mechanics equation, the equations for Omega1 and Omega2 are added.
A1 = 2 * G * ufl.inner(epsilon(u1), epsilon(v1)) * ufl.dx(1) + lame_lambda * ufl.inner(ufl.div(u1), ufl.div(v1)) * ufl.dx(1)
A2 = 2 * G * ufl.inner(epsilon(u2), epsilon(v2)) * ufl.dx(2) + lame_lambda * ufl.inner(ufl.div(u2), ufl.div(v2)) * ufl.dx(2)
B1 = gamma * ufl.inner(epsilon((u1 - u1_n) / dt), epsilon(v1)) * ufl.dx(1) + gamma * ufl.inner(ufl.div((u1 - u1_n) / dt), ufl.div(v1)) * ufl.dx(1)
B2 = gamma * ufl.inner(epsilon((u2 - u2_n) / dt), epsilon(v2)) * ufl.dx(2) + gamma * ufl.inner(ufl.div((u2 - u2_n) / dt), ufl.div(v2)) * ufl.dx(2)
coupling_term1 = -alpha * ufl.inner(p1, ufl.div(v1)) * ufl.dx(1)
coupling_term2 = -alpha * ufl.inner(p2, ufl.div(v2)) * ufl.dx(2)
coupling_term_fracture = ufl.inner(pc("+"), v_jump_norm) * dS
source_mechanics1 = ufl.inner(f, v1) * ufl.dx(1)
source_mechanics2 = ufl.inner(f, v2) * ufl.dx(2)
mechanics_equation = A1 + A2 + B1 + B2 + coupling_term1 + coupling_term2 + coupling_term_fracture - source_mechanics1 - source_mechanics2

# Mass balance in the matrix, the equations for Omega1 and Omega2 are added.
accumulation1 = ufl.inner(c_0 * ((p1 - p1_n) / dt) + alpha * ufl.div((u1 - u1_n) / dt), q1) * ufl.dx(1)
accumulation2 = ufl.inner(c_0 * ((p2 - p2_n) / dt) + alpha * ufl.div((u2 - u2_n) / dt), q2) * ufl.dx(2)
flux1 = k / mu * ufl.inner(ufl.grad(p1), ufl.grad(q1)) * ufl.dx(1)
flux2 = k / mu * ufl.inner(ufl.grad(p2), ufl.grad(q2)) * ufl.dx(2)
f_fluid = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow1 = ufl.inner(f_fluid, q1) * ufl.dx(1)
source_flow2 = ufl.inner(f_fluid, q2) * ufl.dx(2)
matrix_fracture_coupling = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p1("+")),q1("+")) * dS + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p2("-")),q2("-")) * dS
fluid_equation = accumulation1 + accumulation2 + flux1 + flux2 - source_flow1 - source_flow2 - matrix_fracture_coupling

# Mass balance in the fracture
tang = ufl.as_vector((1.0, 0.0))  # Tangent vector along the fracture, used to compute the tangential gradient along the fracture.
accumulation_fracture = ufl.inner(c_f * g(x) * ((pc("+") - pc_n("+")) / dt) - ufl.inner(((u1("+") - u1_n("+") - u2("-") + u2_n("-")) / dt), n("+")), qc("+")) * dS
flux_fracture = 1 / (12 * mu) * ufl.inner(g(x)**3 * ufl.dot(ufl.grad(pc("+")), tang), ufl.dot(ufl.grad(qc("+")), tang)) * dS
f_fluid_fracture = fem.Constant(mesh, default_scalar_type(0))  # Mass source term
source_flow_fracture = ufl.inner(g(x) * f_fluid_fracture, qc("+")) * dS
matrix_fracture_coupling_2 = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p1("+")), qc("+")) * dS + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p2("-")), qc("+")) * dS
fracture_fluid_equation = accumulation_fracture + flux_fracture - source_flow_fracture + matrix_fracture_coupling_2

# Full equation system
F = [mechanics_equation, fluid_equation, fracture_fluid_equation]

# Initial condition
def initial_condition_pressure(x):
    return 0 * x[0]

def initial_condition_displacement(x):
    return (0 * x[0], 0 * x[0])
u1_n.interpolate(initial_condition_displacement)
p1_n.interpolate(initial_condition_pressure)
u2_n.interpolate(initial_condition_displacement)
p2_n.interpolate(initial_condition_pressure)
pc_n.interpolate(initial_condition_pressure)

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
    F, (u1, u2, p1, p2, pc), bcs=bc,
    petsc_options_prefix="mixed_dimensional_", petsc_options=petsc_options,
    kind="mpi", restriction=restriction
)

# Time-stepping loop
step = 0
u1times = []
u2times = []
p1times = []
p2times = []
pctimes = []
u1times.append(u1_n)
u2times.append(u2_n)
p1times.append(p1_n)
p2times.append(p2_n)
pctimes.append(pc_n)
for _ in range(num_steps):
    t += dt
    u1, u2, p1, p2, pc = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    print(converged_reason)
    assert converged_reason > 0
    u1_n.x.array[:] = u1.x.array  # Update solution at previous time step
    u2_n.x.array[:] = u2.x.array
    p1_n.x.array[:] = p1.x.array  # Update solution at previous time step
    p2_n.x.array[:] = p2.x.array
    pc_n.x.array[:] = pc.x.array  # Update solution at previous time step
    u1times.append(u1_n)
    u2times.append(u2_n)
    p1times.append(p1_n)
    p2times.append(p2_n)
    pctimes.append(pc_n)
    step += 1