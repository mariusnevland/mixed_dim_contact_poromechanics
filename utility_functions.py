import gmsh
import dolfinx
import mpi4py.MPI
import matplotlib.pyplot as plt
import numpy as np
import ufl

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


def ordered_interior_entities(facets, mesh, subdomains):

    # Ensure consistent ordering of "+" and "-" sides for neighboring cells of the interface
    # containing the fracture.
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


def measure_away_from_tips(mesh, subdomains, boundaries_and_interfaces):

    # Create custom integration measures that exclude cells too close to the fracture tips.
    tdim = mesh.topology.dim
    tip_left  = np.array([0.3, 0.5])
    tip_right = np.array([0.7, 0.5])
    cells_plus = subdomains.indices[subdomains.values == 1]
    cells_minus = subdomains.indices[subdomains.values == 2]
    midpoints_plus = dolfinx.mesh.compute_midpoints(
        mesh, tdim, cells_plus
    )
    midpoints_minus = dolfinx.mesh.compute_midpoints(
        mesh, tdim, cells_minus
    )
    delta = 0.05

    xmid_plus = midpoints_plus[:, 0]
    ymid_plus = midpoints_plus[:, 1]
    xmid_minus = midpoints_minus[:, 0]
    ymid_minus = midpoints_minus[:, 1]

    r_left_plus = np.sqrt(
        (xmid_plus - tip_left[0])**2
        + (ymid_plus - tip_left[1])**2
    )
    r_right_plus = np.sqrt(
        (xmid_plus - tip_right[0])**2
        + (ymid_plus - tip_right[1])**2
    )
    r_left_minus = np.sqrt(
        (xmid_minus - tip_left[0])**2
        + (ymid_minus - tip_left[1])**2
    )
    r_right_minus = np.sqrt(
        (xmid_minus - tip_right[0])**2
        + (ymid_minus - tip_right[1])**2
    )
    mask_plus = (
        (r_left_plus > delta)
        & (r_right_plus > delta)
    )
    mask_minus = (
        (r_left_minus > delta)
        & (r_right_minus > delta)
    )
    cells_test_plus = cells_plus[mask_plus].astype(np.int32)
    cells_test_minus = cells_minus[mask_minus].astype(np.int32)
    dx_frac_away = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_data=[
            (1, cells_test_plus),
            (2, cells_test_minus)
        ]
    )

    fdim = mesh.topology.dim - 1
    # Fracture facets
    fracture_facets = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 6]

    # Compute facet midpoints
    facet_midpoints = dolfinx.mesh.compute_midpoints(
        mesh,
        fdim,
        fracture_facets
    )
    xmid = facet_midpoints[:, 0]

    mask_frac = (
        (xmid > 0.3 + delta)
        & (xmid < 0.7 - delta)
    )
    fracture_facets_away = fracture_facets[mask_frac].astype(np.int32)
    fracture_facets_away = ordered_interior_entities(fracture_facets_away, mesh, subdomains)
    dS_frac_away = ufl.Measure(
        "dS",
        domain=mesh,
        subdomain_data=[
            (1, fracture_facets_away)
        ]
    )
    return dx_frac_away, dS_frac_away


def draw_multiple_loglog_slopes(
    fig,
    ax,
    origin,
    triangle_width,
    slopes,
    dashed_extra_slopes=False,
    inverted=False,
    color=None,
    label=True,
    labelcolor=None,
):
    """This function draws slopes or "convergence triangles" into loglog plots.

    References:
        The creation of the main triangle is from:
        https://gist.github.com/w1th0utnam3/a0189dc8a2c067ccb56e1de8c317b190

        All other functionality (insertion of extra slope lines, adaptation of label
        locations etc.) are inspired from the original reference.

    Parameters:
        fig: The figure.
        ax: The axes object to draw to.
        origin: The origin coordinates of the triangle.
        triangle_width: The width in inches of the triangle.
        slopes: The list of slopes to be drawn. That is, orders of convergence for the
            "convergence triangle(s)".
        dashed_extra_slopes: Bool value of whether the non-max slopes should be dashed
        inverted: Whether to mirror the triangle (if the 90 degree angle of the
            triangle is in the lower right or upper left).
        color: Color of the triangle edges.
        label: Whether to enable labeling of the slopes. Defaults to True.
        labelcolor: The color of the slope labels. Defaults to edge color.

    Example:
        # Create a figure with plotted array values
            fig, ax = plt.subplots()
            ax.loglog(x, y)

        # Then call this function using fig, ax and other parameters:
            draw_multiple_loglog_slopes(
                fig,
                ax,
                origin=(x[-1], y_disp[-1]),
                triangle_width=1.0,
                slopes=[1, 2],
                labelcolor=(0.33, 0.33, 0.33),
            )
            plt.show()
    """

    polygon_kwargs = {}
    label_kwargs = {}
    zorder = 10
    alpha = 0.25

    if color is not None:
        polygon_kwargs["color"] = color
    else:
        polygon_kwargs["color"] = (0.25, 0.25, 0.25)
        color = (0.25, 0.25, 0.25)

    if labelcolor is not None:
        label_kwargs["color"] = labelcolor
    else:
        label_kwargs["color"] = polygon_kwargs["color"]

    polygon_kwargs["linewidth"] = 0.8 * plt.rcParams["lines.linewidth"]
    label_kwargs["fontsize"] = 1.2 * plt.rcParams["font.size"]

    if inverted:
        triangle_width = -triangle_width

    # Convert the origin into figure coordinates in inches
    origin_disp = ax.transData.transform(origin)
    origin_dpi = fig.dpi_scale_trans.inverted().transform(origin_disp)

    # Obtain the bottom-right corner in data coordinates
    corner_dpi = origin_dpi + triangle_width * np.array([1.0, 0.0])
    corner_disp = fig.dpi_scale_trans.transform(corner_dpi)
    corner = ax.transData.inverted().transform(corner_disp)

    (x1, y1) = (origin[0], origin[1])
    x2 = corner[0]

    # The width of the triangle in data coordinates
    width = x2 - x1

    # Compute offset of the slope
    log_offset = y1 / (x1 ** max(slopes))

    y2 = log_offset * ((x1 + width) ** max(slopes))

    # The vertices of the slope
    a = origin
    b = corner
    c = [x2, y2]

    # Draw the slope triangle
    X = np.array([a, b, c])
    triangle = plt.Polygon(
        X[:3, :], alpha=alpha, fill=True, zorder=zorder, **polygon_kwargs
    )
    ax.add_patch(triangle)

    # Convert vertices into display space
    a_disp = ax.transData.transform(a)
    b_disp = ax.transData.transform(b)
    c_disp = ax.transData.transform(c)

    # Figure out the center of the triangle sides in display space
    bottom_center_disp = a_disp + 0.5 * (b_disp - a_disp)
    bottom_center = ax.transData.inverted().transform(bottom_center_disp)

    right_center_disp = b_disp + 0.5 * (c_disp - b_disp)
    right_center = ax.transData.inverted().transform(right_center_disp)

    # Label alignment depending on inversion parameter and whether or not the slope is
    # negative or positive
    if np.any(slopes > np.zeros(len(slopes))):
        va_xlabel = "top" if not inverted else "bottom"
        ha_ylabel = "left" if not inverted else "right"

        # Label offset depending on inversion parameter
        offset_xlabel = (
            [0.0, -0.33 * label_kwargs["fontsize"]]
            if not inverted
            else [0.0, 0.33 * label_kwargs["fontsize"]]
        )
        offset_ylabel = (
            [0.33 * label_kwargs["fontsize"], 0.0]
            if not inverted
            else [-0.33 * label_kwargs["fontsize"], 0.0]
        )
    elif np.any(slopes < np.zeros(len(slopes))):
        va_xlabel = "bottom" if not inverted else "top"
        ha_ylabel = "right" if not inverted else "left"

        # Label offset depending on inversion parameter
        offset_xlabel = (
            [0.0, 0.33 * label_kwargs["fontsize"]]
            if not inverted
            else [0.0, -0.33 * label_kwargs["fontsize"]]
        )
        offset_ylabel = (
            [-0.33 * label_kwargs["fontsize"], 0.0]
            if not inverted
            else [0.33 * label_kwargs["fontsize"], 0.0]
        )

    # Draw the slope labels
    ax.annotate(
        "$1$",
        bottom_center,
        xytext=offset_xlabel,
        textcoords="offset points",
        ha="center",
        va=va_xlabel,
        zorder=zorder,
        **label_kwargs,
    )
    if label:
        if not inverted:
            label_point = [
                x2 + 0.02 * x2,
                y2 - 0.03 * y2,
            ]  # Upper corner if not inverted
            label_va = "center"
            ha = "left"
            if len(slopes) == 1:
                label_point[1] = (y2 + X[1][1]) / 2
                label_va = "center"
        else:
            label_point = [x2 - 0.02 * x2, y2]  # Lower corner if inverted
            label_va = "center"
            ha = "right"
            if len(slopes) == 1:
                label_point[1] = (y2 + X[1][1]) / 2
                label_va = "top"

        ax.annotate(
            f"${abs(max(slopes))}$",
            label_point,
            xytext=[0, 0],
            textcoords="offset points",
            ha=ha,
            va=label_va,
            zorder=zorder,
            **label_kwargs,
        )

    # Draw dashed slope labels
    if len(slopes) > 1:
        # Find the maximum slope value
        max_slope = max(slopes)

        # Create a new list excluding all occurrences of the maximum slope value
        dashed_slopes = [slope for slope in slopes if slope != max_slope]
        for slope in dashed_slopes:
            dashed_log_offset = y1 / (x1**slope)
            dashed_y2 = dashed_log_offset * ((x1 + width) ** slope)

            # Draw the (potentially) dashed slope lines
            if dashed_extra_slopes:
                linestyle = "--"
            else:
                linestyle = "-"
            ax.plot(
                [x1, x2],
                [y1, dashed_y2],
                linestyle=linestyle,
                linewidth=0.99,
                color=color,
                zorder=zorder - 2,
                alpha=alpha,
            )

            if inverted:
                dashed_label_point = [x2, dashed_y2]
                dashed_label_point[0] = label_point[0]

                label_va = "center"
                label_ha = "right"
            else:
                dashed_label_point = [x2, dashed_y2]
                dashed_label_point[0] = label_point[0]
                label_va = "center"

                label_ha = "left"

            # Draw slope labels
            ax.annotate(
                f"${slope}$",
                dashed_label_point,
                xytext=[0, 0],
                textcoords="offset points",
                ha=label_ha,
                va=label_va,
                zorder=zorder - 2,  # Lower zorder value
                **label_kwargs,
            )