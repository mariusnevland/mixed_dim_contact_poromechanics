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
mesh_size = 1. / 8.

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
pfr1 = gmsh.model.geo.addPoint(0.3, 0.5, 0.0, mesh_size)
pfr2 = gmsh.model.geo.addPoint(0.7, 0.5, 0.0, mesh_size)
midpoint_left = gmsh.model.geo.addPoint(0.0, 0.5, 0.0, mesh_size)
midpoint_right = gmsh.model.geo.addPoint(1.0, 0.5, 0.0, mesh_size)
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
# External boundary minus the bottom
gmsh.model.addPhysicalGroup(1, [lower_boundary_to_midpoint_right, midpoint_right_to_upper_boundary, l3, upper_boundary_to_midpoint_left, midpoint_left_to_lower_boundary], 1)
# Bottom boundary
gmsh.model.addPhysicalGroup(1, [l1], 4)
# Fracture and interfaces between matrix subdomains
gmsh.model.addPhysicalGroup(1, [mid_to_fracture_left, fracture_to_mid_right], 2)
gmsh.model.addPhysicalGroup(1, [frac], 3)
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
facets_partial_omega = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 1]
facets_bottom = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 4]
facets_interface_not_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 2]
facets_frac = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 3]

# Define associated measures
dx = ufl.Measure("dx", subdomain_data=subdomains)

# plotter = DolfinxPlotter.plot_mesh_tags(mesh, boundaries_and_interfaces, "boundaries and interfaces")
# plotter.show(screenshot=f"mesh_boundaries_and_interfaces_test.png")

# Define a custom measure on the fracture/interfaces to ensure a consistent ordering between + and - sides.
integration_entities_on_gamma = dolfinx.fem.compute_integration_domains(
    dolfinx.fem.IntegralType.interior_facet, mesh.topology, facets_interface_not_frac)
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


# Function spaces
V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))  # Space for the displacement
Q = fem.functionspace(mesh, ("Lagrange", 1))  # Space for the matrix pressure
Qc = Q.clone()  # Space for the fracture pressure
(u, p, pc) = (fem.Function(V), fem.Function(Q), fem.Function(Qc))  # Solutions
(v, q, qc) = (ufl.TestFunction(V), ufl.TestFunction(Q), ufl.TestFunction(Qc))  # Test functions
(u_n, p_n, pc_n) = (fem.Function(V), fem.Function(Q), fem.Function(Qc))  # Solutions at previous timestep

# Define restrictions
fdim = boundaries_and_interfaces.dim
dofs_frac = fem.locate_dofs_topological(Qc, fdim, facets_frac)
dofs_Q_omega = fem.locate_dofs_topological(Q, subdomains.dim, cells_omega)
dofs_V_omega = fem.locate_dofs_topological(V, subdomains.dim, cells_omega)
restriction_Qc_frac = multiphenicsx.fem.DofMapRestriction(Qc.dofmap, dofs_frac)
restriction_Q = multiphenicsx.fem.DofMapRestriction(Q.dofmap, dofs_Q_omega)
restriction_V = multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V_omega)
restriction = [restriction_V, restriction_Q]

# Define boundary conditions
zero = dolfinx.fem.Constant(mesh, default_scalar_type(0))
zero_vector = dolfinx.fem.Constant(mesh, default_scalar_type((0,) * mesh.geometry.dim))
u_D = fem.Function(V)
p_D = fem.Function(Q)
u_D.x.array[:] = 0
p_D.x.array[:] = 0
dofs_V = fem.locate_dofs_topological(V, fdim, facets_partial_omega)
dofs_Q = fem.locate_dofs_topological(Q, fdim, facets_partial_omega)
bc_u = fem.dirichletbc(zero_vector, dofs_V, V)
bc_p = fem.dirichletbc(zero, dofs_Q, Q)
bc = [bc_u, bc_p]

# Material parameters
lame_lambda = 1.0  # Lamé's first parameter
G = 1.0  # Shear modulus (Lamé's second parameter)
gamma = 1.0  # Viscoelastic parameter
alpha = 1.0  # Biot coefficient
c_0 = 1.0  # Specific storage coefficient
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

def gap(x):
    return 0 * x[0]  # Simple, flat obstacle.

def indicator_function(x):
    # Indicator function, used to distinguish between open and closed cells.
    # Equals 1 if x>0, 0 if x<0 and 1/2 if x=0.
    return 0.5 * (ufl.sign(x) + 1) 


# Mechanics equation
f = fem.Constant(mesh, default_scalar_type((0, -1)))  # Source term (body force)
a = 2 * G * ufl.inner(epsilon(u), epsilon(v)) * ufl.dx + lame_lambda * ufl.inner(ufl.div(u), ufl.div(v)) * ufl.dx
b = gamma * ufl.inner(epsilon((u - u_n) / dt), epsilon(v)) * ufl.dx + gamma * ufl.inner(ufl.div((u - u_n) / dt), ufl.div(v)) * ufl.dx
coupling_term = -alpha * ufl.inner(p, ufl.div(v)) * ufl.dx
source_mechanics = ufl.inner(f, v) * ufl.dx
mechanics_equation = a + b + coupling_term - source_mechanics

# Flow equation in matrix
accumulation = ufl.inner(c_0 * ((p - p_n) / dt) + alpha * ufl.div((u - u_n) / dt), q) * ufl.dx
flux = k / mu * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
f_fluid = fem.Constant(mesh, default_scalar_type(1))  # Mass source term
source_flow = ufl.inner(f_fluid, q) * ufl.dx
fluid_equation = accumulation + flux - source_flow

# Flow equation in fracture

# Full equation system
F = [mechanics_equation, fluid_equation]

# Initial condition
def initial_condition_pressure(x):
    return 0 * x[0]

def initial_condition_displacement(x):
    return (0 * x[0], 0 * x[0])
u_n.interpolate(initial_condition_displacement)
p_n.interpolate(initial_condition_pressure)

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
    F, (u, p), bcs=bc,
    petsc_options_prefix="mixed_dimensional_", petsc_options=petsc_options,
    kind="mpi", restriction=restriction
)

# Time-stepping loop
step = 0
utimes = []
ptimes = []
utimes.append(u_n)
ptimes.append(p_n)
for _ in range(num_steps):
    t += dt
    u, p = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    print(converged_reason)
    assert converged_reason > 0
    u_n.x.array[:] = u.x.array  # Update solution at previous time step
    p_n.x.array[:] = p.x.array  # Update solution at previous time step
    utimes.append(u_n)
    ptimes.append(p_n)
    step += 1


# Plotting
uplot = utimes[-1]
pplot = ptimes[-1]
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(uplot.function_space)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_plotter = pv.Plotter()

# Map pressure dofs to the same ordering as displacement dofs, so that they can be plotted together. 
u_coords = uplot.function_space.tabulate_dof_coordinates()
p_coords = pplot.function_space.tabulate_dof_coordinates()
u_idx = np.lexsort(u_coords.T)
p_idx = np.lexsort(p_coords.T)
mapped_p = np.zeros_like(pplot.x.array)
mapped_p[u_idx] = pplot.x.array[p_idx]

# Pad displacement vector with zero z-component
u = uplot.x.array.reshape((-1, mesh.geometry.dim)).copy()
u_3d = np.zeros((u.shape[0], 3))
u_3d[:, :2] = u
u_grid.point_data["u"] = u_3d
warped = u_grid.warp_by_vector("u", factor=2)

warped.point_data["p"] = mapped_p
u_plotter.add_mesh(
    warped,
    scalars="p",            
    show_edges=False,
    scalar_bar_args={"title": "Pressure", "vertical": True,
                        "title_font_size": 24, "n_labels": 6,
                        "position_x": 0.85, "position_y": 0.103, "height": 0.815},
    clim=[0, 0.196]
)
u_plotter.view_xy()
u_plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
u_plotter.show(screenshot=f"biot_no_friction.png")