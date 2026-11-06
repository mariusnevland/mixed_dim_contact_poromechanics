from dolfinx import plot, fem, default_real_type, default_scalar_type
import numpy as np
import ufl
import pyvista as pv
pv.off_screen = True
from basix.ufl import element, mixed_element
from matplotlib import pyplot as plt
from utility_functions import *
import multiphenicsx.fem
import multiphenicsx.fem.petsc

def solve_mixed_dimensional_model(mesh, subdomains, boundaries_and_interfaces, final_time, 
                     num_steps, penalty=1, plot_sol=False):

    # Time stepping parameters
    t = 0.0
    T = final_time
    dt = T / num_steps

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

    entities_interface = ordered_interior_entities(
        facets_interface_not_frac, mesh, subdomains
    )

    entities_fracture = ordered_interior_entities(
        facets_frac, mesh, subdomains
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
    u_D1.x.array[:] = 0
    u_D2.x.array[:] = 0
    p_D1.x.array[:] = 0
    p_D2.x.array[:] = 0
    dofs_V2 = fem.locate_dofs_topological((V2_0, V2), fdim, facets_bottom)
    dofs_Q2 = fem.locate_dofs_topological((Q2_0, Q2), fdim, facets_bottom)
    bc_u2 = fem.dirichletbc(u_D2, dofs_V2, V2_0)
    bc_p2 = fem.dirichletbc(p_D2, dofs_Q2, Q2_0)
    bc = [bc_u2, bc_p2]

    # Material parameters
    lame_lambda = 1.0  # Lamé's first parameter
    G = 1.0  # Shear modulus (Lamé's second parameter)
    gamma = 1.0  # Viscoelastic parameter
    alpha = 1.0  # Biot coefficient
    c_0 = 1.0  # Specific storage coefficient
    c_f = 1.0  # Fluid compressibility
    mu = 0.0001  # Fluid viscosity
    k = 1.0  # Permeability
    mu_fr = 1.0  # Friction coefficient

    # Useful functions
    def epsilon(u):
        return ufl.sym(
            ufl.grad(u)
        )  # Strain tensor

    fracture_gap = 1e-4
    def g(x):  # Initial fracture gap.
        return 2 * fracture_gap * x[1]

    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    # Contact condition
    pen = penalty  # Penalty/stiffness parameter
    u_jump_norm = ufl.inner(n("+"), u1("+") - u2("-"))  # Normal jump of displacement across the fracture
    u_jump_tang = u1("+") - u2("-") - u_jump_norm * n("+")  # Tangential jump of displacement across the fracture
    v_jump_norm = ufl.inner(n("+"), v1("+") - v2("-"))
    v_jump_tang = v1("+") - v2("-") - v_jump_norm * n("+")
    u_jump_tang_prev = u1_n("+") - u2_n("-") - ufl.inner(n("+"), u1_n("+") - u2_n("-")) * n("+")

    penetration = u_jump_norm - g(x)

    positive_penetration = ufl.conditional(
        ufl.gt(penetration, 0.0),
        penetration,
        0.0
    )
    norm_traction = -pen * positive_penetration
    trial = pen * (
    u_jump_tang - u_jump_tang_prev
    )

    trial_sq = ufl.inner(trial, trial)

    friction_bound = -mu_fr * norm_traction
    # >= 0 because norm_traction <= 0

    zero_vec = ufl.as_vector(
        tuple(0.0 for _ in range(mesh.geometry.dim))
    )

    tang_traction = ufl.conditional(

        # No normal contact -> no friction
        ufl.le(friction_bound, 0.0),

        zero_vec,

        # Contact active
        ufl.conditional(

            # Stick
            ufl.le(
                trial_sq,
                friction_bound**2
            ),

            trial,

            # Slip
            friction_bound
            * trial
            / ufl.sqrt(trial_sq)
        )
    )
    contact_condition = norm_traction * v_jump_norm * dS_frac + ufl.inner(tang_traction, v_jump_tang) * dS_frac   # Frictional contact condition

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
    trac = fem.Constant(mesh, default_scalar_type((0.001, -0.1)))
    neumann = ufl.inner(trac, v1) * ds(4)
    interface_term = ufl.inner(lu("+"), v1("+") - v2("-")) * dS_intf + ufl.inner(mv("+"), u1("+") - u2("-")) * dS_intf
    mechanics_equation = A1 + A2 + B1 + B2 + coupling_term1 + coupling_term2 + coupling_term_fracture - source_mechanics1 - source_mechanics2 - interface_term - contact_condition - neumann

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
    f_fluid_fracture = fem.Constant(mesh, default_scalar_type(1.5e2))  # Mass source term
    source_flow_fracture = ufl.inner(g(x) * f_fluid_fracture, qc("+")) * dS_frac
    matrix_fracture_coupling_2 = 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p1("+")), qc("+")) * dS_frac + 1 / (6 * mu) * ufl.inner(g(x) * (pc("+") - p2("-")), qc("+")) * dS_frac
    fracture_fluid_equation = accumulation_fracture + flux_fracture - source_flow_fracture + matrix_fracture_coupling_2

    # Full equation system
    F = mechanics_equation + fluid_equation + fracture_fluid_equation

    # Initial condition
    def initial_condition_pressure(x):
        return 0 * x[1]
    def initial_condition_displacement(x):
        return (0 * x[1], 0 * x[1])
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
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": 2,
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_max_it": 100,
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_7": 4
    }

    w.x.array[:] = w_n.x.array
    problem = multiphenicsx.fem.petsc.NonlinearProblem(
        [F], [w], bcs=bc,
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
    u1times.append(w.sub(0).collapse())
    u2times.append(w.sub(1).collapse())
    p1times.append(w.sub(3).collapse())
    p2times.append(w.sub(4).collapse())
    pctimes.append(w.sub(6).collapse())
    for _ in range(num_steps):
        t += dt
        _ = problem.solve()
        converged_reason = problem.solver.getConvergedReason()
        assert converged_reason > 0
        w_n.x.array[:] = w.x.array  # Update solution at previous time step
        u1times.append(w.sub(0).collapse())
        u2times.append(w.sub(1).collapse())
        p1times.append(w.sub(3).collapse())
        p2times.append(w.sub(4).collapse())
        pctimes.append(w.sub(6).collapse())
        step += 1


    # # Plotting
    if plot_sol:
        u1plot = u1times[-1]
        u2plot = u2times[-1]
        pplot1 = p1times[-1]
        pplot2 = p2times[-1]
        # Locate displacement DOF blocks on the fracture
        blocks_u1_frac = fem.locate_dofs_topological(
            u1plot.function_space,
            fdim,
            facets_frac
        )
        blocks_u2_frac = fem.locate_dofs_topological(
            u2plot.function_space,
            fdim,
            facets_frac
        )
        bs1 = u1plot.function_space.dofmap.index_map_bs
        bs2 = u2plot.function_space.dofmap.index_map_bs

        # Expand block indices to scalar component indices
        dofs_u1_frac = (
            bs1 * blocks_u1_frac[:, None] + np.arange(bs1)
        ).reshape(-1)

        dofs_u2_frac = (
            bs2 * blocks_u2_frac[:, None] + np.arange(bs2)
        ).reshape(-1)

        # # Reshape vector values
        u1_vals = u1plot.x.array[dofs_u1_frac].reshape(-1, bs1)
        u2_vals = u2plot.x.array[dofs_u2_frac].reshape(-1, bs2)
        u1_prev = u1times[-2]
        u2_prev = u2times[-2]
        u1_old_vals = u1_prev.x.array.reshape(-1, bs1)[blocks_u1_frac]
        u2_old_vals = u2_prev.x.array.reshape(-1, bs2)[blocks_u2_frac]
        n_plus = np.array([0.0, -1.0])

        # Compute slip tendency, i.e., the ratio of tangential traction magnitude to friction bound.
        jump = u1_vals - u2_vals
        jump_old = u1_old_vals - u2_old_vals
        jump_norm = jump @ n_plus
        jump_norm_old = jump_old @ n_plus
        jump_tang_vec = (
            jump
            - jump_norm[:, None] * n_plus
        )
        jump_tang_old_vec = (
            jump_old
            - jump_norm_old[:, None] * n_plus
        )
        trial = pen * (
        jump_tang_vec
        - jump_tang_old_vec
        )
        norm_trial = np.linalg.norm(trial, axis=1)
        coords1 = V1.tabulate_dof_coordinates()[blocks_u1_frac]
        order1 = np.argsort(coords1[:, 0])
        coords1 = coords1[order1]
        g_values = 2 * fracture_gap * coords1[:, 1]
        penetration = jump_norm - g_values
        norm_traction = (
            -pen
            * np.maximum(penetration, 0.0)
        )
        friction_bound = (
            -mu_fr * norm_traction
        )
        tang_traction = np.zeros_like(trial)
        contact = friction_bound > 0.0
        stick = (
            contact
            & (norm_trial <= friction_bound)
        )
        slip = (
            contact
            & (norm_trial > friction_bound)
        )
        tang_traction[stick] = trial[stick]
        tang_traction[slip] = (
            friction_bound[slip, None]
            * trial[slip]
            / norm_trial[slip, None]
        )
        ratio = np.zeros_like(friction_bound)
        mask = friction_bound > 0
        ratio[mask] = (
            np.linalg.norm(tang_traction[mask], axis=1)
            / friction_bound[mask]
        )
        _, ax = plt.subplots()
        ax.plot(np.arange(1, len(ratio)+1), ratio, color="blue", zorder=2)
        ax.set_xlabel("Fracture cell number", fontsize=14)
        ax.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
        ax.set_ylabel("Slip tendency", fontsize=14)
        plt.savefig("slip_tendency.png")

        # Plot the pressure field in the matrix, choose arbitrarily to plot the + side on the interface nodes.
        plotter = pv.Plotter()
        p_topology, p_cell_types, p_geometry = plot.vtk_mesh(pplot2.function_space)
        p_grid = pv.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
        pplot2_modified = fem.Function(pplot2.function_space)
        pplot2_modified.x.array[:] = pplot2.x.array
        dofs_intf_frac = fem.locate_dofs_topological(
                pplot2.function_space,
                fdim,
                np.concatenate([facets_frac, facets_interface_not_frac])
        )
        pplot2_modified.x.array[dofs_intf_frac] = 0.0
        p_combined = fem.Function(pplot1.function_space)
        p_combined.x.array[:] = pplot1.x.array + pplot2_modified.x.array
        p_grid.point_data["p"] = p_combined.x.array
        plotter.add_mesh(
            p_grid, show_edges=False, cmap="coolwarm",
            scalar_bar_args={"title": r"$p$", "vertical": True,
                        "title_font_size": 24, "n_labels": 6,
                        "position_x": 0.8, "position_y": 0.103, "height": 0.815}
        )
        plotter.view_xy()
        plotter.show(screenshot=f"matrix_pressure.png")

    return V1, V2, Q1, Q2, Qc, u1times, u2times, p1times, p2times, pctimes
