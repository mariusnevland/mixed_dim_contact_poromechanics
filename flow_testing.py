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

from test_interface import V1
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
fracx_left = 0.0
fracx_right = 1.0
fracy = 0.5
midpoint_left = gmsh.model.geo.addPoint(0.0, fracy, 0.0, mesh_size)
midpoint_right = gmsh.model.geo.addPoint(1.0, fracy, 0.0, mesh_size)
upper_boundary_to_midpoint_left = gmsh.model.geo.addLine(pul, midpoint_left)
midpoint_right_to_upper_boundary = gmsh.model.geo.addLine(midpoint_right, pur)
midpoint_left_to_lower_boundary = gmsh.model.geo.addLine(midpoint_left, pll)
lower_boundary_to_midpoint_right = gmsh.model.geo.addLine(plr, midpoint_right)
interface = gmsh.model.geo.addLine(midpoint_left, midpoint_right)

# Matrix subdomains, split into a positive and negative side relative to the fracture
upper_boundary = gmsh.model.geo.addCurveLoop([upper_boundary_to_midpoint_left, interface, midpoint_right_to_upper_boundary, l3])
upper_rectangle = gmsh.model.geo.addPlaneSurface([upper_boundary])
lower_boundary = gmsh.model.geo.addCurveLoop([midpoint_left_to_lower_boundary, l1, lower_boundary_to_midpoint_right, -interface])
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
# Interface between matrix subdomains
gmsh.model.addPhysicalGroup(1, [interface], 5)

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
facets_interface = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 5]

# Define associated measures
dx = ufl.Measure("dx", subdomain_data=subdomains)
ds = ufl.Measure("ds", subdomain_data=boundaries_and_interfaces)

# Define a custom measure on the fracture facets to ensure a consistent ordering between + and - sides.
integration_entities_on_gamma = dolfinx.fem.compute_integration_domains(
    dolfinx.fem.IntegralType.interior_facet, mesh.topology, facets_interface)
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

# Mixed function spaces
P1_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
W = fem.functionspace(mesh, mixed_element([P1_pressure, P1_pressure]))
Q1_0 = W.sub(0)
Q2_0 = W.sub(1)
Q1, map_Q1 = Q1_0.collapse()
Q2, map_Q2 = Q2_0.collapse()
w = fem.Function(W)  # Current solution     
w_n = fem.Function(W)  # Solution at previous timestep
(p1, p2) = ufl.split(w)
(p1_n, p2_n) = ufl.split(w_n)
(q1, q2) = ufl.TestFunctions(W)


# Define restrictions
fdim = boundaries_and_interfaces.dim
dofs_Q1_Omega1 = map_Q1[fem.locate_dofs_topological(Q1, subdomains.dim, cells_Omega1)]
dofs_Q2_Omega2 = map_Q2[fem.locate_dofs_topological(Q2, subdomains.dim, cells_Omega2)]
dofs_restricted = np.unique(
    np.concatenate([
        dofs_Q1_Omega1,
        dofs_Q2_Omega2,
    ])
).astype(np.int32)
restriction_mixed = multiphenicsx.fem.DofMapRestriction(W.dofmap, dofs_restricted)
restriction = [restriction_mixed]

# Boundary conditions
p_D1 = fem.Function(Q1)
p_D2 = fem.Function(Q2)
p_D1.x.array[:] = 0
p_D2.x.array[:] = 0
dofs_Q1 = fem.locate_dofs_topological((Q1_0, Q1), fdim, facets_top)
dofs_Q2 = fem.locate_dofs_topological((Q2_0, Q2), fdim, facets_bottom)
bc_p1 = fem.dirichletbc(p_D1, dofs_Q1, Q1_0)
bc_p2 = fem.dirichletbc(p_D2, dofs_Q2, Q2_0)
bc = [bc_p1, bc_p2]

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

x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
# Mass balance in the matrix, the equations for Omega1 and Omega2 are added.
accumulation1 = ufl.inner(c_0 * ((p1 - p1_n) / dt), q1) * dx(1)
accumulation2 = ufl.inner(c_0 * ((p2 - p2_n) / dt), q2) * dx(2)
flux1 = k / mu * ufl.inner(ufl.grad(p1), ufl.grad(q1)) * dx(1)
flux2 = k / mu * ufl.inner(ufl.grad(p2), ufl.grad(q2)) * dx(2)
f_fluid = fem.Constant(mesh, default_scalar_type(1))  # Mass source term
source_flow1 = ufl.inner(f_fluid, q1) * dx(1)
source_flow2 = ufl.inner(f_fluid, q2) * dx(2)
fluid_equation = accumulation1 + accumulation2 + flux1 + flux2 - source_flow1 - source_flow2

# Full equation system
F = fluid_equation
# Initial condition
def initial_condition_pressure(x):
    return 0 * x[0]
w_n.sub(0).interpolate(initial_condition_pressure)
w_n.sub(1).interpolate(initial_condition_pressure)

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
p1times = []
p2times = []
lagtimes = []
p1times.append(w.sub(0).collapse())
p2times.append(w.sub(1).collapse())
for _ in range(num_steps):
    t += dt
    _ = problem.solve()
    converged_reason = problem.solver.getConvergedReason()
    print(converged_reason)
    assert converged_reason > 0
    w_n.x.array[:] = w.x.array  # Update solution at previous time step
    p1times.append(w.sub(0).collapse())
    p2times.append(w.sub(1).collapse())
    step += 1


# # Plotting
pplot1 = p1times[-1]
pplot2 = p2times[-1]
print(np.max(pplot1.x.array))
print(np.max(pplot2.x.array))
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