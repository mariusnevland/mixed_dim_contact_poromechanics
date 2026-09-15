import gmsh
import dolfinx
import mpi4py.MPI

# Create the coarsest mesh used in the spatial convergence analysis.
def create_initial_mesh(mesh_size):

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

    return mesh, subdomains, boundaries_and_interfaces


def refine_mesh_with_tags(mesh, subdomains, facet_tags):

    refined_mesh, parent_cell, parent_facet = dolfinx.mesh.refine(
        mesh,
        option=dolfinx.mesh.RefinementOption.parent_cell_and_facet
    )
    refined_mesh.topology.create_connectivity(refined_mesh.topology.dim - 1, refined_mesh.topology.dim)
    refined_mesh.topology.create_connectivity(refined_mesh.topology.dim, refined_mesh.topology.dim - 1)
    refined_mesh.topology.create_connectivity(refined_mesh.topology.dim, refined_mesh.topology.dim)

    refined_subdomains = dolfinx.mesh.transfer_meshtag(
        subdomains,
        refined_mesh,
        parent_cell
    )

    refined_facet_tags = dolfinx.mesh.transfer_meshtag(
        facet_tags,
        refined_mesh,
        parent_cell,
        parent_facet
    )

    return refined_mesh, refined_subdomains, refined_facet_tags


def create_initial_nonfracture_mesh(mesh_size):

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

    # Interface
    midpoint_left = gmsh.model.geo.addPoint(0.0, 0.5, 0.0, mesh_size)
    midpoint_right = gmsh.model.geo.addPoint(1.0, 0.5, 0.0, mesh_size)
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

    return mesh, subdomains, boundaries_and_interfaces