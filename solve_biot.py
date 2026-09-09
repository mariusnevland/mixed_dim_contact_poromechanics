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

def solve_biot(mesh, subdomains, boundaries_and_interfaces, final_time, num_steps):

    # Time stepping parameters
    t = 0.0
    T = final_time
    dt = T / num_steps

    # Create connectivities required by the rest of the code
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    cells_Omega1 = subdomains.indices[subdomains.values == 1]
    cells_Omega2 = subdomains.indices[subdomains.values == 2]
    facets_left = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 1]
    facets_right = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 2]
    facets_bottom = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 3]
    facets_top = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 4]
    facets_boundary = np.concatenate([facets_left, facets_right, facets_bottom, facets_top])
    facets_interface = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 5]

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
        facets_interface
    )
    dS_data = [
        (5, entities_interface),
    ]

    dS_all = ufl.Measure(
        "dS",
        domain=mesh,
        subdomain_data=dS_data
    )
    dS_intf = dS_all(5)

    # Mixed function spaces
    P1_displacement = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type)
    P1_pressure = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    W = fem.functionspace(mesh, mixed_element([P1_displacement, P1_displacement, P1_displacement, P1_pressure, P1_pressure, P1_pressure]))
    V1_0 = W.sub(0)
    V2_0 = W.sub(1)
    MV_0 = W.sub(2)  # Lagrange multiplier space for displacement
    Q1_0 = W.sub(3)
    Q2_0 = W.sub(4)
    MQ_0 = W.sub(5)  # Lagrange multiplier space for pressure
    V1, map_V1 = V1_0.collapse()
    V2, map_V2 = V2_0.collapse()
    MV, map_MV = MV_0.collapse()
    Q1, map_Q1 = Q1_0.collapse()
    Q2, map_Q2 = Q2_0.collapse()
    MQ, map_MQ = MQ_0.collapse()
    w = fem.Function(W)  # Current solution     
    w_n = fem.Function(W)  # Solution at previous timestep
    (u1, u2, lu, p1, p2, lp) = ufl.split(w)
    (u1_n, u2_n, lu_n, p1_n, p2_n, lp_n) = ufl.split(w_n)
    (v1, v2, mv, q1, q2, mq) = ufl.TestFunctions(W)


    # Define restrictions
    fdim = boundaries_and_interfaces.dim
    dofs_Q1_Omega1 = map_Q1[fem.locate_dofs_topological(Q1, subdomains.dim, cells_Omega1)]
    dofs_Q2_Omega2 = map_Q2[fem.locate_dofs_topological(Q2, subdomains.dim, cells_Omega2)]
    dofs_MQ_Gamma = map_MQ[fem.locate_dofs_topological(MQ, fdim, facets_interface)]
    # Locate vector DOF BLOCKS on Omega1
    blocks_V1_Omega1 = fem.locate_dofs_topological(
        V1,
        subdomains.dim,
        cells_Omega1
    )
    # Locate vector DOF BLOCKS on Omega2
    blocks_V2_Omega2 = fem.locate_dofs_topological(
        V2,
        subdomains.dim,
        cells_Omega2
    )
    blocks_M_Gamma = fem.locate_dofs_topological(
        MV,
        fdim,
        facets_interface
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

    scalar_M_Gamma = (
        bs3 * blocks_M_Gamma[:, None]
        + np.arange(bs3)
    ).reshape(-1)
    # Now map collapsed scalar DOFs -> mixed-space scalar DOFs
    dofs_V1_Omega1 = map_V1[scalar_V1_Omega1]
    dofs_V2_Omega2 = map_V2[scalar_V2_Omega2]
    dofs_MV_Gamma = map_MV[scalar_M_Gamma]
    dofs_restricted = np.unique(
        np.concatenate([
            dofs_V1_Omega1,
            dofs_V2_Omega2,
            dofs_MV_Gamma,
            dofs_Q1_Omega1,
            dofs_Q2_Omega2,
            dofs_MQ_Gamma,
        ])
    ).astype(np.int32)
    restriction_mixed = multiphenicsx.fem.DofMapRestriction(W.dofmap, dofs_restricted)
    restriction = [restriction_mixed]

    # Boundary conditions
    u_D1 = fem.Function(V1)
    u_D2 = fem.Function(V2)
    p_D1 = fem.Function(Q1)
    p_D2 = fem.Function(Q2)
    u_D1.x.array[:] = 0
    u_D2.x.array[:] = 0
    p_D1.x.array[:] = 0
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

    # Useful funcions
    def epsilon(u):
        return ufl.sym(
            ufl.grad(u)
        )  # Strain tensor
    
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    # Mass balance in the matrix
    accumulation1 = ufl.inner(c_0 * ((p1 - p1_n) / dt) + alpha * ufl.div((u1 - u1_n) / dt), q1) * dx(1)
    accumulation2 = ufl.inner(c_0 * ((p2 - p2_n) / dt) + alpha * ufl.div((u2 - u2_n) / dt), q2) * dx(2)
    flux1 = k / mu * ufl.inner(ufl.grad(p1), ufl.grad(q1)) * dx(1)
    flux2 = k / mu * ufl.inner(ufl.grad(p2), ufl.grad(q2)) * dx(2)
    f_fluid = fem.Constant(mesh, default_scalar_type(0))
    source_flow1 = ufl.inner(f_fluid, q1) * dx(1)
    source_flow2 = ufl.inner(f_fluid, q2) * dx(2)
    interface_term = ufl.inner(lp("+"), q1("+") - q2("-")) * dS_intf + ufl.inner(mq("+"), p1("+") - p2("-")) * dS_intf
    fluid_equation = accumulation1 + accumulation2 + flux1 + flux2 - source_flow1 - source_flow2 + interface_term

    f = fem.Constant(mesh, default_scalar_type((0.0, -1.0)))
    # Mechanics equation, the equations for Omega1 and Omega2 are added.
    A1 = 2 * G * ufl.inner(epsilon(u1), epsilon(v1)) * dx(1) + lame_lambda * ufl.inner(ufl.div(u1), ufl.div(v1)) * dx(1)
    A2 = 2 * G * ufl.inner(epsilon(u2), epsilon(v2)) * dx(2) + lame_lambda * ufl.inner(ufl.div(u2), ufl.div(v2)) * dx(2)
    B1 = gamma * ufl.inner(epsilon((u1 - u1_n) / dt), epsilon(v1)) * dx(1) + gamma * ufl.inner(ufl.div((u1 - u1_n) / dt), ufl.div(v1)) * dx(1)
    B2 = gamma * ufl.inner(epsilon((u2 - u2_n) / dt), epsilon(v2)) * dx(2) + gamma * ufl.inner(ufl.div((u2 - u2_n) / dt), ufl.div(v2)) * dx(2)
    coupling_term1 = -alpha * ufl.inner(p1, ufl.div(v1)) * dx(1)
    coupling_term2 = -alpha * ufl.inner(p2, ufl.div(v2)) * dx(2)
    source_mechanics1 = ufl.inner(f, v1) * dx(1)
    source_mechanics2 = ufl.inner(f, v2) * dx(2)
    interface_term = ufl.inner(lu("+"), v1("+") - v2("-")) * dS_intf + ufl.inner(mv("+"), u1("+") - u2("-")) * dS_intf
    mechanics_equation = A1 + A2 + B1 + B2 + coupling_term1 + coupling_term2 - source_mechanics1 - source_mechanics2 - interface_term


    # Full equation system
    F = fluid_equation + mechanics_equation
    # Initial condition
    def initial_condition_displacement(x):
        return (0 * x[0], 0 * x[0])
    def initial_condition_pressure(x):
        return 0 * x[0]
    w_n.sub(0).interpolate(initial_condition_displacement)
    w_n.sub(1).interpolate(initial_condition_displacement)
    w_n.sub(2).interpolate(initial_condition_displacement)
    w_n.sub(3).interpolate(initial_condition_pressure)
    w_n.sub(4).interpolate(initial_condition_pressure)
    w_n.sub(5).interpolate(initial_condition_pressure)

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
        step += 1

    # # Plotting
    u1plot = u1times[-1]
    u2plot = u2times[-1]

    plotter = pv.Plotter()
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u1plot.function_space)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u1plot.x.array.reshape((-1, 2))
    plotter.add_mesh(
        u_grid, scalars="u", component=0, show_edges=True
    )
    plotter.view_xy()
    plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
    plotter.show(screenshot=f"mixed_biot_disp_upper.png")

    plotter = pv.Plotter()
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u2plot.function_space)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u2plot.x.array.reshape((-1, 2))
    plotter.add_mesh(
        u_grid, scalars="u", component=0, show_edges=True
    )
    plotter.view_xy()
    plotter.show_bounds(xtitle=" ", ytitle=" ", bounds=(0, 1, 0, 1, 0, 0))
    plotter.show(screenshot=f"mixed_biot_disp_lower.png")

    return V1, V2, Q1, Q2, u1times, u2times, p1times, p2times
