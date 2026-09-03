"""
generate_tiles.py
Generates watertight STL files for helmet lattice tile designs.

Can be run directly (generates the initial 5 designs) or imported
by surrogate.py to auto-generate ML-recommended designs.

Requirements:
    pip install numpy scikit-image trimesh scipy

Output:
    STL files written to ../stl_output/
"""

import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import functools
import math
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "stl_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_X = 50.8   # 2 in
TILE_Y = 50.8   # 2 in
TILE_Z = 38.1   # 1.5 in

# Marching-cubes sample spacing (mm). Resolution is DERIVED from physical size so a
# small tile -> a small grid. Grid points = (X/pitch)*(Y/pitch)*(Z/pitch); at 0.5 mm
# this 2x2x1.5 in prism is ~0.8 M points vs. the old 400^3 = 64 M that buffered/OOM'd.
VOXEL_PITCH_MM = 0.5
RESOLUTION = round(TILE_X / VOXEL_PITCH_MM)   # ~102 samples across the 2 in span

# Wall/strut lattices (re-entrant, honeycomb, Voronoi) have ~1.2-2 mm features, i.e.
# only 2-4 voxels per wall at 0.5 mm. Sample those finer so the walls don't come out
# lumpy or locally pinched. 145^2 x 109 ~ 2.3 M points -- still cheap.
FINE_PITCH_MM = 0.35
RESOLUTION_FINE = round(TILE_X / FINE_PITCH_MM)

TPU_DENSITY_G_MM3 = 1.21e-3
TPU_Es_MPA        = 25.0
TILE_AREA_MM2     = TILE_X * TILE_Y
TARGET_FORCE_N    = 7100.0
MIN_WALL_MM       = 1.2


# ── Whole-cell fitting ───────────────────────────────────────────────────────

def _as3(cell):
    """Accept a scalar cell size or a per-axis (cx, cy, cz) tuple -> 3 floats."""
    if np.isscalar(cell):
        return float(cell), float(cell), float(cell)
    cx, cy, cz = cell
    return float(cx), float(cy), float(cz)


def snap_cell_sizes(cell_size, tx, ty, tz):
    """Round to a whole number of cells on each axis and resize the cell to
    tile_dim / n, so the tile holds only complete cells (none sliced at a face).

    `cell_size` may be a scalar (isotropic) or a per-axis (cx, cy, cz) tuple
    (anisotropic / non-cubic cells). Returns ((cx, cy, cz), (nx, ny, nz)). Because
    the period divides each tile extent an integer number of times, the surface
    lands at the same phase on opposite faces -> whole, mirror-tileable cells. For
    TPMS, relative density is set by the isovalue (not the period), so this leaves
    the target density intact.
    """
    cx, cy, cz = _as3(cell_size)
    nx = max(1, round(tx / cx))
    ny = max(1, round(ty / cy))
    nz = max(1, round(tz / cz))
    return (tx / nx, ty / ny, tz / nz), (nx, ny, nz)


# ── SDF functions ──────────────────────────────────────────────────────────────

def gyroid(x, y, z, cell_size):
    cx, cy, cz = _as3(cell_size)
    px, py, pz = 2*np.pi/cx, 2*np.pi/cy, 2*np.pi/cz
    return (np.sin(px*x)*np.cos(py*y) +
            np.sin(py*y)*np.cos(pz*z) +
            np.sin(pz*z)*np.cos(px*x))

def schwartz_p(x, y, z, cell_size):
    cx, cy, cz = _as3(cell_size)
    px, py, pz = 2*np.pi/cx, 2*np.pi/cy, 2*np.pi/cz
    return np.cos(px*x) + np.cos(py*y) + np.cos(pz*z)

def diamond(x, y, z, cell_size):
    cx, cy, cz = _as3(cell_size)
    px, py, pz = 2*np.pi/cx, 2*np.pi/cy, 2*np.pi/cz
    return (np.sin(px*x)*np.sin(py*y)*np.sin(pz*z) +
            np.sin(px*x)*np.cos(py*y)*np.cos(pz*z) +
            np.cos(px*x)*np.sin(py*y)*np.cos(pz*z) +
            np.cos(px*x)*np.cos(py*y)*np.sin(pz*z))

TOPOLOGY_FN = {
    "gyroid":    gyroid,
    "schwartz":  schwartz_p,
    "diamond":   diamond,
}

# The TPMS subset of the blend basis. Kept as its own name because these three are
# the smooth, zero-centred fields; the full generative basis spans all nine
# registered families and is defined with FIELD_TOPOLOGIES further down.
TPMS_BASIS = {"gyroid": gyroid, "schwartz": schwartz_p, "diamond": diamond}


@functools.lru_cache(maxsize=8192)
def _measure_cached(wkey, cx, cy, cz, rho, res):
    """Measure surface-area-to-volume of a (blended) TPMS straight from a small SDF
    grid — the real CAD geometry, not a per-topology constant. Cached on rounded
    args so the optimizer can call it per candidate cheaply. Returns sa_to_v (1/mm).

    Uses a 3-period tile per axis so the value is intensive (cell-shape dependent,
    tile-size independent). Density is hit via the quantile isovalue.
    """
    weights = dict(wkey)
    tx, ty, tz = 3.0 * cx, 3.0 * cy, 3.0 * cz
    x, y, z = np.mgrid[0:tx:res*1j, 0:ty:res*1j, 0:tz:res*1j]
    F = blended_field(x, y, z, (cx, cy, cz), weights)
    iso = float(np.quantile(F, float(np.clip(rho, 0.02, 0.98))))
    try:
        verts, faces, _, _ = measure.marching_cubes(F, iso)
    except Exception:
        return None
    vox = np.array([tx / (res - 1), ty / (res - 1), tz / (res - 1)])
    m = trimesh.Trimesh(verts * vox, faces, process=False)
    return float(m.area) / float(tx * ty * tz)


def measure_tpms_descriptors(weights, cell_size, relative_density, res=32):
    """sa_to_v (1/mm) measured from the actual (blended) TPMS field. weights is a
    blend dict (e.g. {'gyroid':1.0}); cell_size scalar or (cx,cy,cz)."""
    cx, cy, cz = _as3(cell_size)
    w = normalize_blend(weights)
    wkey = tuple(sorted((k, round(v, 3)) for k, v in w.items()))
    sav = _measure_cached(wkey, round(cx, 2), round(cy, 2), round(cz, 2),
                          round(float(relative_density), 3), int(res))
    return None if sav is None else {"sa_to_v": sav}

def density_to_isovalue(relative_density):
    return -0.8 + (relative_density - 0.15) * 4.0


# ── Watertight TPMS mesh generation ───────────────────────────────────────────

def generate_tile(topology_fn, cell_size, relative_density,
                  tz=TILE_Z, res=RESOLUTION, iso_mode="linear", iso_value=None):
    """
    Generates a watertight TPMS lattice tile.

    Key fix: the SDF grid is padded with values ABOVE the isovalue on all
    6 faces. This tells marching_cubes that everything outside the tile is
    void, so it generates closing faces at every tile boundary edge.
    Without this, the mesh is open where walls intersect the tile edge,
    causing 'non-manifold' errors in slicers.

    iso_mode: "linear" uses the calibrated density->isovalue map (original
    behavior for pure gyroid/schwartz). "quantile" picks the isovalue so the
    solid voxel fraction == relative_density exactly — needed for BLENDED fields
    whose density<->isovalue relationship isn't the single-topology one.
    """
    # Snap the requested cell size to a whole number of cells on each axis so the
    # tile has no half cells sliced at a face.
    cells, cell_counts = snap_cell_sizes(cell_size, TILE_X, TILE_Y, tz)

    # res = samples across the X span; derive Y/Z counts to keep ~uniform pitch
    # (so the short Z axis isn't oversampled).
    nx = ny = res
    nz = max(8, round(res * tz / TILE_X))
    x, y, z = np.mgrid[0:TILE_X:nx*1j,
                        0:TILE_Y:ny*1j,
                        0:tz:nz*1j]

    sdf = topology_fn(x, y, z, cells)
    if iso_value is not None:
        iso = float(iso_value)            # caller-driven (density correction loop)
    elif iso_mode == "quantile":
        # Solid = {sdf < iso}; choose iso at the rho-quantile so solid fraction == rho.
        iso = float(np.quantile(sdf, float(np.clip(relative_density, 0.02, 0.98))))
    else:
        iso = density_to_isovalue(relative_density)

    # Pad all 6 faces with a value clearly ABOVE the isosurface ("definitely void")
    # so marching_cubes closes the mesh at every tile boundary edge.
    #
    # Pad FAR above, not just +2: marching cubes puts the closing face where the
    # field crosses `iso` on its way from the last real sample to the pad, so a
    # modest pad value places that face a noticeable fraction of a voxel OUTSIDE the
    # tile (tiles measured 51.2 mm instead of 50.8). A large pad drives the crossing
    # back onto the boundary sample plane, where it belongs.
    lo, hi = float(np.min(sdf)), float(np.max(sdf))
    pad_value = hi + 20.0 * (hi - lo + 1.0)
    sdf_padded = np.pad(sdf, 1, mode='constant', constant_values=pad_value)

    verts, faces, _, _ = measure.marching_cubes(sdf_padded, iso)

    # Convert padded voxel coordinates back to mm.
    # Padding shifted origin by 1 voxel; spacing = extent / (samples - 1) because
    # mgrid's n*1j form is INCLUSIVE of both endpoints (n samples -> n-1 gaps).
    # Dividing by n instead shrank every tile by ~1% (50.8 -> 50.3 mm).
    voxel_size = np.array([TILE_X / (nx - 1), TILE_Y / (ny - 1), tz / (nz - 1)])
    verts = (verts - 1.0) * voxel_size

    mesh = trimesh.Trimesh(verts, faces)
    mesh = _repair(mesh)
    mesh.metadata["cell_sizes_mm"] = tuple(round(c, 3) for c in cells)
    mesh.metadata["cell_counts"] = cell_counts
    # For distance-to-skeleton fields the isovalue IS the half-wall thickness in mm,
    # so keep it -- printability_report() turns it into a real wall measurement.
    mesh.metadata["iso"] = float(iso)

    expected_mass_g = relative_density * (TILE_X * TILE_Y * tz) * TPU_DENSITY_G_MM3
    return mesh, expected_mass_g


def generate_blended_tile(weights, cell_size, relative_density, tz=TILE_Z, res=None,
                          field_kw=None, cap=None, correct_density=True,
                          tol=0.004, max_iter=3, drop_islands=True):
    """GENERATIVE tile: a convex blend of ANY registered families -> a new in-between
    geometry. `weights` is a dict like {"gyroid":0.4,"voronoi":0.35,"elytra":0.25}.

    Density is hit exactly via the quantile isovalue plus the same meshed-volume
    correction loop build_tile() uses -- a blend's density<->iso curve is its own and
    is not predictable from the members'. `cap` (e.g. MAX_FAMILY_WEIGHT) additionally
    enforces a per-family ceiling so the result can never collapse toward one family.
    """
    w = normalize_blend(weights, cap=cap)
    fn = lambda gx, gy, gz, cs: blended_field(gx, gy, gz, cs, w, field_kw=field_kw)
    if res is None:
        # Any wall/strut family in the mix contributes ~1.2-2 mm features, so sample
        # at the fine pitch or those walls come out lumpy.
        res = RESOLUTION_FINE if any(k in _FINE for k in w) else RESOLUTION

    mesh, iso = _mesh_to_density(fn, cell_size, relative_density, tz=tz, res=res,
                                 iso_mode="quantile", distance=True,
                                 correct_density=correct_density,
                                 tol=tol, max_iter=max_iter)
    if drop_islands:
        mesh = _drop_islands(mesh)

    tile_vol = TILE_X * TILE_Y * tz
    mesh.metadata["topology"] = "blend"
    mesh.metadata["blend"] = {k: round(float(v), 3) for k, v in w.items()}
    mesh.metadata["iso"] = iso
    mesh.metadata["rel_density_designed"] = float(relative_density)
    mesh.metadata["rel_density_as_built"] = float(mesh.volume) / tile_vol
    # A blend is support-free only if every contributing family is: a weighted sum of
    # z-invariant fields is still z-invariant, but one z-varying member breaks it.
    mesh.metadata["support_free_by_construction"] = all(
        FIELD_TOPOLOGIES[k]["support_free"] for k in w)
    mass = mesh.metadata["rel_density_as_built"] * tile_vol * TPU_DENSITY_G_MM3
    return mesh, mass


def generate_graded_tile(cell_size=8.0, rho_soft=0.18, rho_stiff=0.35, res=RESOLUTION):
    """Two-layer graded gyroid: soft outer (head side) + stiff inner (shell side)."""
    half_z = TILE_Z / 2
    soft,  m1 = generate_tile(gyroid, cell_size, rho_soft,  tz=half_z, res=res)
    stiff, m2 = generate_tile(gyroid, cell_size, rho_stiff, tz=half_z, res=res)
    stiff.apply_translation([0, 0, half_z])
    merged = trimesh.util.concatenate([soft, stiff])
    merged = _repair(merged)
    # Each half is snapped to whole cells over half_z; total Z cells = 2 * (per half).
    cells, counts = snap_cell_sizes(cell_size, TILE_X, TILE_Y, half_z)
    merged.metadata["cell_sizes_mm"] = tuple(round(c, 3) for c in cells)
    merged.metadata["cell_counts"] = (counts[0], counts[1], counts[2] * 2)
    return merged, m1 + m2


def generate_stacked_tile(layers, res=None):
    """Alternating / hybrid tile: stack different TPMS layers along Z into one
    watertight body (head side = first layer).

    layers : list of (topology, rel_density, cell_size_mm, thickness_mm). Topology
             must be any name in FIELD_TOPOLOGIES (TPMS, re-entrant, honeycomb,
             elytra, voronoi, spinodoid); each layer snaps to whole cells over its
             own thickness. generate_graded_tile is the 2-layer gyroid special case.
             Mixing families is allowed and interesting -- e.g. a soft spinodoid
             head-side layer under a stiff re-entrant shell-side layer.
    Returns (mesh, expected_mass_g).
    """
    parts, z0, mass = [], 0.0, 0.0
    for topo, rho, cell, th in layers:
        if topo not in FIELD_TOPOLOGIES:
            raise ValueError(f"layer topology '{topo}' not scriptable here; "
                             f"choose from {sorted(FIELD_TOPOLOGIES)}")
        m, mm = build_tile(topo, cell, rho, tz=th, res=res)
        m.apply_translation([0, 0, z0])
        parts.append(m)
        z0 += th
        mass += mm
    merged = trimesh.util.concatenate(parts)
    merged = _repair(merged)
    # cell_size per layer may be a scalar or a per-axis (cx, cy, cz) tuple
    # (anisotropic cells), so round element-wise to avoid round(tuple).
    def _rc(c):
        return tuple(round(float(v), 3) for v in c) if isinstance(c, (tuple, list)) \
            else round(float(c), 3)
    merged.metadata["layers"] = [(t, round(r, 3), _rc(c), round(h, 3))
                                 for (t, r, c, h) in layers]
    merged.metadata["cell_sizes_mm"] = tuple(_rc(c) for (_, _, c, _) in layers)
    merged.metadata["cell_counts"] = len(layers)
    return merged, mass


def generate_rotating_squares(tile_x=TILE_X, tile_y=TILE_Y, tile_z=TILE_Z,
                               cell=10.0, rel_density=0.29, res=RESOLUTION):
    """
    Perforated rotating-squares auxetic tile — support-free (all walls vertical,
    zero overhangs). The printed material is a solid sheet with a grid of square
    HOLES; the thin ligaments left between holes act as the auxetic hinges. This is
    the inverse of the rigid-squares-on-hinges solid: the squares are the *removed*
    material, not what is left behind.

    cell        = nominal cell pitch (mm); snapped to whole cells per axis
    rel_density = target printed relative density (sheet minus square holes)
    """
    # Snap the cell pitch to a whole number of cells per axis so every hole is
    # complete (no hole sliced by a tile face).
    ncx = max(1, round(tile_x / cell))
    ncy = max(1, round(tile_y / cell))
    ax, ay = tile_x / ncx, tile_y / ncy

    # Size the square holes from the SNAPPED pitch to hit the target density:
    # printed ρ* = 1 - s²/(ax·ay)  ->  s = sqrt((1-ρ*)·ax·ay).
    hole_frac = min(0.95, max(0.0, 1.0 - rel_density))
    s = math.sqrt(hole_frac * ax * ay)
    # Keep a printable ligament (>=1.2mm) between holes on the tighter axis.
    s = min(s, min(ax, ay) - 1.2)
    assert s >= 2.0, (f"snapped pitch {min(ax, ay):.1f}mm leaves hole s={s:.1f}mm "
                      f"(<2mm) once a 1.2mm ligament is kept — use a smaller cell")

    xi = np.linspace(0, tile_x, res)
    yi = np.linspace(0, tile_y, res)
    XX, YY = np.meshgrid(xi, yi, indexing='ij')

    holes_2d = np.zeros((res, res), dtype=float)

    # Whole square holes, each centred inside its cell.
    for i in range(ncx):
        for j in range(ncy):
            cx, cy = (i + 0.5) * ax, (j + 0.5) * ay
            holes_2d[(np.abs(XX - cx) <= s/2) & (np.abs(YY - cy) <= s/2)] = 1.0

    holes_smooth = gaussian_filter(holes_2d, sigma=0.8)

    # Field: holes = high (void), sheet = low (solid). Pad all 6 faces with 1.0
    # (void) so the sheet closes at the tile boundary and each square becomes an
    # interior through-hole.
    nz = max(10, int(res * tile_z / tile_x))
    layer = holes_smooth[:, :, np.newaxis]
    vox_3d = np.repeat(layer, nz, axis=2)
    vox_3d = np.pad(vox_3d, 1, mode='constant', constant_values=1.0)

    verts, faces, _, _ = measure.marching_cubes(vox_3d, 0.5)

    # Shift back by 1 voxel in all 3 axes (padding added 1 voxel at each face origin).
    # linspace/repeat are endpoint-inclusive -> n-1 gaps across the span (see generate_tile).
    vox_size = np.array([tile_x, tile_y, tile_z]) / np.array([res - 1, res - 1, nz - 1],
                                                             dtype=float)
    verts = (verts - 1.0) * vox_size

    mesh = trimesh.Trimesh(verts, faces)
    mesh = _repair(mesh)
    mesh.metadata["cell_sizes_mm"] = (round(ax, 3), round(ay, 3))
    mesh.metadata["cell_counts"] = (ncx, ncy)

    rel_density = float(1.0 - holes_2d.mean())
    expected_mass = rel_density * (tile_x * tile_y * tile_z) * TPU_DENSITY_G_MM3
    return mesh, expected_mass, rel_density


# ── Support-free field library: auxetic (re-entrant) + bio-inspired ───────────
#
# Every topology below is written as a field f(x, y, z, cell_size) with the SAME
# convention as the TPMS functions -- SOLID = {f < iso} -- so they all flow through
# generate_tile()'s padding / marching-cubes / repair path unchanged, and hit their
# target relative density exactly via iso_mode="quantile".
#
# PRINTABILITY: all defaults here are PRISMATIC (the field ignores z), so every wall
# is vertical and the overhang angle is exactly 0 -> no supports, which matters
# because support interfaces in TPU 95A are miserable to remove without tearing the
# lattice. The 3D variants (spinodoid mode="isotropic") are available but will need
# supports; printability_report() measures the downward-facing area so you can see
# the cost rather than guess at it.

def _prismatic(field2d):
    """Wrap a 2D field f(X, Y, cell, **kw) as a 3D field that ignores z.

    The result is a vertical extrusion: zero overhang everywhere, support-free.
    The 2D field is evaluated once on the (nx, ny) slice and broadcast along z, so
    it costs the same as a 2D computation regardless of tile height.
    """
    @functools.wraps(field2d)
    def wrapped(x, y, z, cell_size, **kw):
        x = np.asarray(x)
        if x.ndim < 3:
            return field2d(x, np.asarray(y), cell_size, **kw)
        F2 = field2d(x[:, :, 0], np.asarray(y)[:, :, 0], cell_size, **kw)
        return np.repeat(F2[:, :, None], x.shape[2], axis=2)
    return wrapped


def _seg_distance_2d(X, Y, segs):
    """Min Euclidean distance from every (X, Y) grid point to a set of 2D segments.

    segs: iterable of (x0, y0, x1, y1). Thresholding this distance field at t/2
    gives a wall network of thickness t with properly rounded (filleted) joints --
    which is what you want in TPU anyway, since sharp interior corners are where
    tear propagation starts.
    """
    best = np.full(X.shape, np.inf)
    for x0, y0, x1, y1 in segs:
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 <= 1e-12:
            d = np.hypot(X - x0, Y - y0)
        else:
            t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0, 1.0)
            d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
        np.minimum(best, d, out=best)
    return best


# ── Hexagonal family: honeycomb (theta > 0) and RE-ENTRANT auxetic (theta < 0) ──
#
# Both are the same topology with one sign flipped, which is the whole auxetic
# story: ribs of length l at angle theta off horizontal, joined by vertical ribs of
# length h. theta > 0 -> ordinary honeycomb, positive Poisson's ratio. theta < 0 ->
# the ribs point back INWARD (bowtie / re-entrant cell), and pulling the cell open
# in y also opens it in x -> NEGATIVE Poisson's ratio. Under a helmet impact that
# means material flows TOWARD the strike instead of squeezing out from under it.
#
# Periods: x-period cx = 2*l*cos(theta), y-period cy = 2*(h + l*sin(theta)).

def _hex_family_segments(cx, cy, theta_deg, x0, x1, y0, y1, phase=0.25):
    """Wall skeleton for the hex/re-entrant family over [x0,x1] x [y0,y1]."""
    th = math.radians(theta_deg)
    l = cx / (2.0 * math.cos(th))
    h = cy / 2.0 - l * math.sin(th)          # from cy = 2*(h + l*sin(th))
    if h <= 0.05 * l:
        # Anisotropic cell asked for a y-period too short to leave a vertical rib.
        h = 0.05 * l
    segs = []
    # Phase-shift the pattern off the tile boundary so no wall is bisected by a
    # face (a half-thickness edge wall would be under the printable minimum and
    # would buckle first, biasing the whole coupon).
    ox, oy = phase * cx, phase * cy
    i0, i1 = int(math.floor((x0 - cx) / cx)) - 1, int(math.ceil((x1 + cx) / cx)) + 1
    j0, j1 = int(math.floor((y0 - cy) / cy)) - 1, int(math.ceil((y1 + cy) / cy)) + 1
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for sx, sy in ((0.0, 0.0), (0.5, 0.5)):          # two sublattices
                nx_ = (i + sx) * cx + ox
                ny_ = (j + sy) * cy + oy
                top = ny_ + h
                segs.append((nx_, ny_, nx_, top))                    # vertical rib
                segs.append((nx_, top, nx_ + cx / 2, ny_ + cy / 2))  # inclined rib
                segs.append((nx_, top, nx_ - cx / 2, ny_ + cy / 2))  # inclined rib
    return segs


def _hex_family_2d(X, Y, cell_size, theta_deg=30.0, phase=0.25):
    cx, cy, _ = _as3(cell_size)
    if np.isscalar(cell_size):
        # Scalar cell -> equal ribs (h = l), the regular member of the family:
        # cy = cx * (1 + sin(theta)) / cos(theta).
        th = math.radians(theta_deg)
        cy = cx * (1.0 + math.sin(th)) / math.cos(th)
    segs = _hex_family_segments(cx, cy, theta_deg,
                                float(X.min()), float(X.max()),
                                float(Y.min()), float(Y.max()), phase=phase)
    return _seg_distance_2d(X, Y, segs)


@_prismatic
def re_entrant_2d(X, Y, cell_size, theta_deg=-30.0, phase=0.25):
    """AUXETIC re-entrant honeycomb (bowtie cells). theta_deg must be negative."""
    if theta_deg >= 0:
        raise ValueError("re-entrant needs theta_deg < 0 (negative = ribs fold "
                         f"inward); got {theta_deg}. Use honeycomb_2d for theta > 0.")
    return _hex_family_2d(X, Y, cell_size, theta_deg=theta_deg, phase=phase)


@_prismatic
def honeycomb_2d(X, Y, cell_size, theta_deg=30.0, phase=0.25):
    """Hexagonal honeycomb -- bee comb. theta_deg=30 gives regular hexagons."""
    if theta_deg <= 0:
        raise ValueError("honeycomb needs theta_deg > 0; use re_entrant_2d for < 0.")
    return _hex_family_2d(X, Y, cell_size, theta_deg=theta_deg, phase=phase)


@_prismatic
def elytra_2d(X, Y, cell_size, theta_deg=30.0, phase=0.25, column_scale=0.55):
    """BEETLE ELYTRA: honeycomb walls + thickened vertical columns at the nodes.

    The elytron (beetle forewing) is a honeycomb core stitched through by trans-layer
    tubular columns; the columns carry the through-thickness crush load while the
    honeycomb walls stabilise them against buckling. Columns land on the hex nodes
    and are made thicker than the walls by scaling their distance field down
    (column_scale < 1 -> the iso-threshold reaches further out from the axis).
    """
    cx, cy, _ = _as3(cell_size)
    if np.isscalar(cell_size):
        th = math.radians(theta_deg)
        cy = cx * (1.0 + math.sin(th)) / math.cos(th)
    walls = _hex_family_2d(X, Y, cell_size, theta_deg=theta_deg, phase=phase)
    # Column axes at the rib junctions (top of each vertical rib).
    th = math.radians(theta_deg)
    l = cx / (2.0 * math.cos(th))
    h = max(cy / 2.0 - l * math.sin(th), 0.05 * l)
    ox, oy = phase * cx, phase * cy
    nodes = []
    i0, i1 = int(math.floor(X.min() / cx)) - 1, int(math.ceil(X.max() / cx)) + 1
    j0, j1 = int(math.floor(Y.min() / cy)) - 1, int(math.ceil(Y.max() / cy)) + 1
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for sx, sy in ((0.0, 0.0), (0.5, 0.5)):
                nodes.append(((i + sx) * cx + ox, (j + sy) * cy + oy + h))
    cols = np.full(X.shape, np.inf)
    for px, py in nodes:
        np.minimum(cols, np.hypot(X - px, Y - py), out=cols)
    return np.minimum(walls, cols * column_scale)


# ── Trabecular bone / cork: prismatic Voronoi foam ────────────────────────────

@_prismatic
def voronoi_2d(X, Y, cell_size, seed=0, irregularity=0.6, extent=None):
    """VORONOI open-cell foam -- cancellous (trabecular) bone, cork, plant parenchyma.

    Seeds start on a regular grid and are jittered by `irregularity` (0 = perfectly
    ordered, 1 = fully random). That knob is the point: it is a continuous dial from
    periodic to stochastic, and it feeds the contract's `irregularity` column
    directly, so the ML model can search order-vs-disorder as a real design axis
    instead of treating "organic" as an unmodellable category.

    Field = d2 - d1 (distance to 2nd-nearest seed minus nearest), which is 0 exactly
    on the Voronoi edges and grows into the cell interiors -> thresholding gives the
    cell walls. Seeds are replicated into the 8 neighbouring images so the pattern is
    periodic across tile faces and coupons can be butted together seamlessly.
    """
    cx, cy, _ = _as3(cell_size)
    ex, ey = extent if extent is not None else (TILE_X, TILE_Y)
    ncx, ncy = max(2, round(ex / cx)), max(2, round(ey / cy))
    ax, ay = ex / ncx, ey / ncy

    rng = np.random.default_rng(int(seed))
    gx, gy = np.meshgrid((np.arange(ncx) + 0.5) * ax,
                         (np.arange(ncy) + 0.5) * ay, indexing="ij")
    j = float(np.clip(irregularity, 0.0, 1.0)) * 0.5
    pts = np.stack([(gx + rng.uniform(-j, j, gx.shape) * ax).ravel(),
                    (gy + rng.uniform(-j, j, gy.shape) * ay).ravel()], axis=1)

    # Periodic images so the wall pattern matches across opposite tile faces.
    tiled = np.concatenate([pts + np.array([dx * ex, dy * ey])
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)

    d = np.hypot(X[..., None] - tiled[:, 0], Y[..., None] - tiled[:, 1])
    two = np.partition(d, 1, axis=-1)[..., :2]
    return two[..., 1] - two[..., 0]


# ── Spinodoid: spinodal-decomposition bone-like stochastic topology ───────────

def spinodoid(x, y, z, cell_size, seed=0, n_waves=64, mode="lamellar",
              cone_deg=15.0, form="sheet", periodic=False, k_radius=2):
    """SPINODOID -- the bicontinuous morphology of spinodal decomposition, which is
    also what trabecular bone and many biological foams actually look like.

    Built as a Gaussian random field: a sum of N plane waves of equal wavelength and
    random phase, thresholded. Non-periodic, smooth, node-free (no stress-
    concentrating strut junctions -- the reason spinodoids beat truss lattices on
    fatigue and on buckling-dominated crush), and every `seed` is a genuinely
    distinct structure, so this is the generator to use when you want a family of
    unique-but-statistically-identical specimens for replicate testing.

    mode selects which cone the wave vectors are drawn from -- this is the anisotropy
    control, and it doubles as the printability control:
      "lamellar" (default) : k within cone_deg of the X axis -> near-parallel vertical
                             sheets. Support-free AND single-body at liner densities,
                             which is why it is the default; strongly anisotropic
                             (stiff across the sheets, compliant along them).
      "columnar"           : k confined to the XY plane -> field is z-invariant ->
                             vertical walls, zero overhang. BUT at rho ~ 0.25 the 2D
                             solid set sits below its connectivity threshold and
                             breaks into several disconnected columns -- fine near
                             rho = 0.4+, not as a low-density liner.
      "isotropic"          : k uniform on the sphere -> true 3D bicontinuous solid.
                             Best mechanical isotropy, but it HAS overhangs and will
                             need supports; check printability_report() first.

    form selects WHICH set the solid is, and at these densities it decides whether the
    part holds together at all:
      "sheet" (default) : solid = {|F| < iso}, a thin shell wrapped around the zero
                          level set. Connected at any density, so this is the form to
                          use for a 0.15-0.35 liner.
      "solid"           : solid = {F < iso}, the classic minority phase. Only
                          percolates near rho = 0.5. Asking for rho = 0.25 gave 28
                          disconnected pillars -- correct behaviour for spinodal
                          decomposition, useless as a liner. Kept because it is the
                          right choice if you ever run a near-50% density study.
    """
    cx, cy, cz = _as3(cell_size)
    rng = np.random.default_rng(int(seed))
    n = int(n_waves)

    if mode == "columnar":
        phi = rng.uniform(0, 2 * np.pi, n)
        dirs = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
    elif mode == "lamellar":
        c = math.cos(math.radians(cone_deg))
        u = rng.uniform(c, 1.0, n)                      # cos(angle) off the X axis
        t = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(np.clip(1 - u * u, 0, None))
        dirs = np.stack([u, r * np.cos(t), r * np.sin(t)], axis=1)
        dirs *= rng.choice([-1.0, 1.0], size=(n, 1))    # symmetric about the origin
    elif mode == "isotropic":
        v = rng.normal(size=(n, 3))
        dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown spinodoid mode '{mode}'; choose columnar, "
                         "lamellar or isotropic")

    # One wavelength per cell, per axis (anisotropic cells stretch the morphology).
    if periodic:
        # A sum of plane waves repeats with the cell only if every wave vector is a
        # RECIPROCAL-LATTICE point, i.e. k = 2*pi*(i/cx, j/cy, l/cz) with i,j,l whole.
        # Continuous directions give an incommensurate (never-repeating) field, which
        # is what made spinodoid tiles look asymmetric and un-patterned. Snapping each
        # direction to the nearest lattice point at radius k_radius keeps the mode's
        # anisotropy (the cone the directions were drawn from) while forcing exact
        # periodicity. k_radius sets how many wavelengths fit per cell: 1 gives only
        # 26 possible directions (too coarse to look like a spinodoid), 2 gives enough
        # variety while keeping features large enough to print.
        lat = np.rint(dirs * float(k_radius)).astype(int)
        # Rounding can collapse a direction onto the origin (k=0 is a constant, not a
        # wave); send those to +-1 on their dominant axis instead of dropping them.
        dead = ~lat.any(axis=1)
        if dead.any():
            axis = np.argmax(np.abs(dirs[dead]), axis=1)
            fix = np.zeros((int(dead.sum()), 3), dtype=int)
            fix[np.arange(len(axis)), axis] = np.where(
                dirs[dead][np.arange(len(axis)), axis] >= 0, 1, -1)
            lat[dead] = fix
        kvec = 2 * np.pi * lat / np.array([cx, cy, cz])
    else:
        kvec = dirs * np.array([2 * np.pi / cx, 2 * np.pi / cy, 2 * np.pi / cz])
    psi = rng.uniform(0, 2 * np.pi, n)

    F = np.zeros(np.shape(x), dtype=float)
    for i in range(n):
        F += np.cos(kvec[i, 0] * x + kvec[i, 1] * y + kvec[i, 2] * z + psi[i])
    F *= math.sqrt(2.0 / n)

    if form == "sheet":
        return np.abs(F)
    if form == "solid":
        return F
    raise ValueError(f"unknown spinodoid form '{form}'; choose sheet or solid")


# ── Bouligand: helicoidal stack (mantis shrimp dactyl club, arthropod cuticle) ──

def bouligand(x, y, z, cell_size, base="honeycomb", n_plies=6, pitch_deg=60.0, **kw):
    """BOULIGAND / helicoidal stack -- mantis shrimp dactyl club, arthropod cuticle.

    Plies of an in-plane-anisotropic pattern, each rotated by `pitch_deg` from the one
    below, so the stiff direction corkscrews through the thickness. A crack or a shear
    wave can never find a straight path -- it is forced to twist as it advances, which
    is why the mantis shrimp's club survives thousands of strikes. That is a SHEAR
    mechanism, which makes this the topology aimed at the ROTATIONAL half of the STAR
    metric rather than the linear half.

    Built as ONE 3D field (each z-slab evaluates the base pattern at its own rotation)
    rather than as separately meshed plies that get concatenated. That matters: meshing
    plies separately leaves them merely touching face-to-face, which came out
    non-watertight and left loose fragments where a rotated pattern got clipped at the
    tile corners. Meshing the whole field at once fuses the plies into a single solid.

    Only `n_plies` distinct rotations exist, so the 2D base field is evaluated
    n_plies times, not once per z-sample.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if base not in FIELD_TOPOLOGIES:
        raise ValueError(f"unknown bouligand base '{base}'")
    fn2 = FIELD_TOPOLOGIES[base]["fn"]
    X2, Y2 = x[:, :, 0], y[:, :, 0]
    zv = z[0, 0, :]
    zmax = float(zv.max()) or 1.0
    px, py = float(X2.max()) / 2.0, float(Y2.max()) / 2.0

    n_plies = int(n_plies)
    ply_of_z = np.clip((zv / zmax * n_plies).astype(int), 0, n_plies - 1)
    F = np.empty(x.shape, dtype=float)
    for p in range(n_plies):
        sel = np.nonzero(ply_of_z == p)[0]
        if sel.size == 0:
            continue
        a = math.radians(p * pitch_deg)
        ca, sa = math.cos(a), math.sin(a)
        dx, dy = X2 - px, Y2 - py
        # The prismatic wrapper takes the 2D path when handed 2D arrays (z unused).
        F2 = fn2(px + ca * dx + sa * dy, py - sa * dx + ca * dy, None, cell_size, **kw)
        F[:, :, sel] = np.asarray(F2)[:, :, None]
    return F


def _drop_islands(mesh, min_volume_frac=0.01):
    """Delete disconnected fragments below `min_volume_frac` of the total volume.

    Clipping a rotated or stochastic pattern at the tile faces can leave slivers that
    touch nothing. They are not a meshing artefact -- they would print as loose bits
    rattling around the plate, or fail mid-print and get dragged into the nozzle.
    Bodies above the threshold are kept, so a deliberately 2-body tile (the graded
    tile is two stacked halves) survives untouched.
    """
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(parts) <= 1:
        return mesh
    vols = np.array([abs(float(p.volume)) for p in parts])
    total = vols.sum()
    if total <= 0:
        return mesh
    keep = [p for p, v in zip(parts, vols) if v / total >= min_volume_frac]
    if not keep or len(keep) == len(parts):
        return mesh
    return trimesh.util.concatenate(keep)


# ── Topology registry ─────────────────────────────────────────────────────────
#
# Every entry is a field usable by generate_tile(). `quantile` marks fields whose
# density<->isovalue relationship is not the calibrated TPMS one (all the new ones),
# `distance` marks distance-to-skeleton fields, where the chosen isovalue IS the
# half-wall-thickness in mm and can be reported directly as a printability number.

FIELD_TOPOLOGIES = {
    "gyroid":     dict(fn=gyroid,        iso="linear",   distance=False, support_free=True),
    "schwartz":   dict(fn=schwartz_p,    iso="linear",   distance=False, support_free=True),
    "diamond":    dict(fn=diamond,       iso="quantile", distance=False, support_free=True),
    "re_entrant": dict(fn=re_entrant_2d, iso="quantile", distance=True,  support_free=True),
    "honeycomb":  dict(fn=honeycomb_2d,  iso="quantile", distance=True,  support_free=True),
    "elytra":     dict(fn=elytra_2d,     iso="quantile", distance=True,  support_free=True),
    "voronoi":    dict(fn=voronoi_2d,    iso="quantile", distance=True,  support_free=True),
    "spinodoid":  dict(fn=spinodoid,     iso="quantile", distance=False, support_free=True),
    "bouligand":  dict(fn=bouligand,     iso="quantile", distance=True,  support_free=False),
}

# Topologies that want the finer sampling grid (thin walls).
_FINE = {"re_entrant", "honeycomb", "elytra", "voronoi"}


def _mesh_to_density(fn, cell_size, rho_target, tz, res, iso_mode, distance,
                     correct_density=True, tol=0.004, max_iter=3):
    """Mesh `fn` and, for quantile isovalues, walk the isovalue until the MESHED
    volume fraction actually equals rho_target. Returns (mesh, iso).

    Shared by build_tile() and generate_blended_tile(): the quantile isovalue only
    makes the fraction of sample POINTS inside the solid equal rho, which matches the
    volume fraction only when walls are several voxels thick. For thin walls it is
    badly biased, so measure the meshed volume and correct.
    """
    tile_vol = TILE_X * TILE_Y * tz
    rho_target = float(rho_target)
    mesh, _ = generate_tile(fn, cell_size, rho_target, tz=tz, res=res,
                            iso_mode=iso_mode)
    iso = float(mesh.metadata["iso"])
    if not (correct_density and iso_mode == "quantile"):
        return mesh, iso

    rho_now = float(mesh.volume) / tile_vol
    prev = None
    for _ in range(int(max_iter)):
        if not np.isfinite(rho_now) or abs(rho_now - rho_target) <= tol:
            break
        if prev is None:
            # Thin-wall model: volume ~ linear in iso through the origin.
            iso_new = iso * (rho_target / max(rho_now, 1e-6))
        else:
            # Secant through the two most recent (iso, rho) samples.
            (i0, r0) = prev
            denom = (rho_now - r0)
            if abs(denom) < 1e-9:
                break
            iso_new = iso + (rho_target - rho_now) * (iso - i0) / denom
        if not np.isfinite(iso_new) or iso_new <= 0 and distance:
            break
        prev = (iso, rho_now)
        trial, _ = generate_tile(fn, cell_size, rho_target, tz=tz, res=res,
                                 iso_value=iso_new)
        rho_trial = float(trial.volume) / tile_vol
        if not np.isfinite(rho_trial) or rho_trial <= 0:
            break
        mesh, iso, rho_now = trial, iso_new, rho_trial
    return mesh, iso


# ── Cross-family field blending (GENERATIVE search space) ─────────────────────
# Any convex mix of the registered implicit fields is itself a valid implicit field,
# so the optimizer can search a continuous simplex over ALL nine families rather than
# picking one label. Two things have to be handled for the weights to mean anything:
#
#   1. SCALE. The TPMS members are trig sums (zero-centred, magnitude ~1) while the
#      wall members are DISTANCE fields in mm whose magnitude scales with cell size.
#      Summing those raw lets whichever field has the larger numeric range set the
#      isosurface regardless of its weight -- a nominal 45/30/25 mix comes out looking
#      ~95% like one family. Each field is therefore rank-normalised onto a common
#      uniform [0,1] scale first, which is distribution-free and so survives the very
#      skewed distance fields.
#   2. ORIENTATION. Every family is meshed as solid = {F < iso} at a quantile
#      isovalue (near-skeleton for distance fields, low-field for TPMS), so "low =
#      solid" holds for all of them and the mixture is coherent rather than fighting.

BLEND_BASIS = list(FIELD_TOPOLOGIES)

# Families whose field repeats EXACTLY with the unit cell, so a tile built from them
# is a true repeating pattern rather than a one-off blob. Verified by comparing the
# field at x against x + cell:
#   gyroid/schwartz/diamond/re_entrant/honeycomb/elytra  exact, no configuration needed
#   voronoi    exact ONLY at irregularity=0 (jittered seeds are what break it; the
#              jitter is periodic across the whole tile but not across a cell)
#   spinodoid  exact ONLY with periodic=True (reciprocal-lattice wave vectors)
#   bouligand  NEVER -- each ply is the base pattern rotated by an arbitrary angle
#              about the tile centre, which does not map the cell lattice onto
#              itself, so no parameter choice makes it cell-periodic.
PERIODIC_FAMILIES = {"gyroid", "schwartz", "diamond", "re_entrant", "honeycomb",
                     "elytra", "voronoi", "spinodoid"}

# Field kwargs required to put a family into its periodic configuration.
PERIODIC_FIELD_KW = {
    "voronoi":   {"irregularity": 0.0},
    "spinodoid": {"periodic": True},
}


def periodic_field_kw(weights, extra=None):
    """Build the field_kw dict that puts every family in `weights` into repeating mode.

    `extra` (e.g. per-family seeds) is merged on top, so a stochastic family stays
    reproducible without losing its periodic configuration.
    """
    kw = {}
    for fam in weights:
        base = dict(PERIODIC_FIELD_KW.get(fam, {}))
        if extra and fam in extra:
            base.update(extra[fam])
        if base:
            kw[fam] = base
    return kw

# No single family may exceed this share of a design. The ceiling implies a FLOOR on
# how many families a design must use: with cap 0.45 you need at least
# ceil(1/0.45) = 3, so "never pure" comes free rather than needing its own rule.
MAX_FAMILY_WEIGHT = 0.45
MIN_FAMILIES = math.ceil(1.0 / MAX_FAMILY_WEIGHT)


def project_capped_simplex(weights, cap=MAX_FAMILY_WEIGHT):
    """L1-normalise `weights` subject to every entry <= cap (water-filling).

    Clip the over-cap entries down to `cap`, spread the freed mass across the others
    in proportion to their remaining headroom, repeat. Redistributing by headroom
    (rather than by value) is what stops the fix from pushing a different family over
    the ceiling, so this converges immediately instead of oscillating.
    """
    names = [k for k in weights if k in FIELD_TOPOLOGIES]
    w = np.array([max(0.0, float(weights[k])) for k in names], dtype=float)
    n = len(names)
    cap = float(cap)
    if n == 0 or w.sum() <= 0:
        raise ValueError("no positive weights over the registered families")
    if n * cap < 1.0 - 1e-9:
        raise ValueError(
            f"{n} famil{'y' if n == 1 else 'ies'} cannot sum to 1 under a per-family "
            f"cap of {cap:g}; need at least {math.ceil(1.0 / cap)}")
    w /= w.sum()
    for _ in range(64):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = np.where(over, 0.0, cap - w)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break
        w += excess * room / total_room
    return {k: float(v) for k, v in zip(names, w)}


def normalize_blend(weights: dict, cap=None) -> dict:
    """Clamp negatives and L1-normalise blend weights over the registered families.

    cap=None leaves the mix unconstrained (any single family may dominate); pass
    MAX_FAMILY_WEIGHT to also enforce the per-family ceiling.
    """
    if cap is not None:
        return {k: v for k, v in project_capped_simplex(weights, cap).items()
                if v > 1e-9}
    w = {k: max(0.0, float(v)) for k, v in weights.items() if k in FIELD_TOPOLOGIES}
    s = sum(w.values())
    if s <= 0:
        return {"gyroid": 1.0}
    return {k: v / s for k, v in w.items() if v > 0}


def _rank_normalize(F):
    """Map an arbitrary field onto uniform [0,1] by rank, order-preserving.

    Distribution-free, so a mm-scaled distance field and a unit-scale trig field come
    out directly comparable and the blend weights -- not the fields' relative
    magnitudes -- decide the mix. Ties are broken stably, which is harmless here
    because the isovalue is itself chosen as a quantile of the blended result.
    """
    flat = np.asarray(F, dtype=float).ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=float)
    ranks[order] = np.arange(flat.size, dtype=float)
    return (ranks / max(flat.size - 1, 1)).reshape(np.shape(F))


def blended_field(x, y, z, cell_size, weights: dict, field_kw=None, cap=None):
    """Convex blend of registered implicit fields -> one new implicit surface.

    field_kw: optional {family: {kwarg: value}} passed through to individual fields
    (seed=/irregularity= for voronoi, mode= for spinodoid, base=/pitch_deg= for
    bouligand), so a stochastic member stays reproducible.
    """
    w = normalize_blend(weights, cap=cap)
    field_kw = field_kw or {}
    F = None
    for name, wt in w.items():
        raw = FIELD_TOPOLOGIES[name]["fn"](x, y, z, cell_size,
                                           **field_kw.get(name, {}))
        term = wt * _rank_normalize(raw)
        F = term if F is None else F + term
    return F


def build_tile(topology, cell_size, relative_density, tz=TILE_Z, res=None,
               correct_density=True, tol=0.004, max_iter=3, drop_islands=True, **kw):
    """Generate any registered topology by NAME, with the right isovalue mode and
    sampling resolution chosen for it. This is the single entry point the ML
    pipeline and main() both use.

    Extra kwargs pass through to the field (seed=, irregularity=, theta_deg=,
    mode=, base=, column_scale=, ...), so a stochastic topology + seed is reproducible.

    correct_density: the quantile isovalue makes the fraction of SAMPLE POINTS inside
    the solid equal rho, which is only the same thing as the volume fraction when the
    walls are several voxels thick. For thin walls it is badly biased -- a re-entrant
    tile asked for rho=0.25 came out at 0.133. So measure the meshed volume and
    correct the isovalue. Wall volume is very nearly linear in the isovalue for a
    distance field, so the first correction usually lands inside tolerance and the
    loop costs one extra meshing pass, not many.

    Returns (mesh, expected_mass_g).
    """
    if topology not in FIELD_TOPOLOGIES:
        raise ValueError(f"unknown topology '{topology}'; "
                         f"choose from {sorted(FIELD_TOPOLOGIES)}")
    spec = FIELD_TOPOLOGIES[topology]
    if res is None:
        res = RESOLUTION_FINE if topology in _FINE else RESOLUTION

    fn = spec["fn"]
    if kw:
        base_fn = fn
        fn = lambda gx, gy, gz, cs: base_fn(gx, gy, gz, cs, **kw)

    tile_vol = TILE_X * TILE_Y * tz
    rho_target = float(relative_density)

    mesh, iso = _mesh_to_density(fn, cell_size, rho_target, tz=tz, res=res,
                                 iso_mode=spec["iso"], distance=spec["distance"],
                                 correct_density=correct_density,
                                 tol=tol, max_iter=max_iter)

    if drop_islands:
        mesh = _drop_islands(mesh)

    mesh.metadata["topology"] = topology
    mesh.metadata["params"] = dict(kw)
    mesh.metadata["iso"] = iso
    mesh.metadata["rel_density_designed"] = rho_target
    mesh.metadata["rel_density_as_built"] = float(mesh.volume) / tile_vol
    if spec["distance"]:
        # For a distance field the isovalue is literally the half-wall thickness.
        mesh.metadata["wall_thickness_mm"] = 2.0 * iso
    mass = mesh.metadata["rel_density_as_built"] * tile_vol * TPU_DENSITY_G_MM3
    return mesh, mass


def generate_bouligand_tile(base="honeycomb", cell_size=8.0, relative_density=0.25,
                            n_plies=6, pitch_deg=60.0, tz=TILE_Z, res=None, **kw):
    """Convenience wrapper -> build_tile("bouligand", ...). See bouligand().

    Printability note: within a ply every wall is vertical, but at each ply interface
    a wall only lands on the ply below where the two rotated patterns cross, and
    bridges between crossings. Small pitch angles make those bridges long, so keep
    pitch_deg >= 45 with a honeycomb base and read the unsupported-area number out of
    printability_report() before committing to a long print.
    """
    if res is None:
        res = RESOLUTION_FINE if base in _FINE else RESOLUTION
    return build_tile("bouligand", cell_size, relative_density, tz=tz, res=res,
                      base=base, n_plies=n_plies, pitch_deg=pitch_deg, **kw)


# ── Printability + as-built measurement ───────────────────────────────────────

def printability_report(mesh, tile_volume_mm3=None, overhang_limit_deg=45.0,
                        min_wall_mm=MIN_WALL_MM, verbose=True):
    """Measure what actually matters before sending a tile to the K1 Max.

    Returns a dict with:
      watertight            : slicer will reject the mesh if False
      rel_density_measured  : mesh volume / tile bounding volume -- the AS-BUILT
                              density, which is the value the handoff contract wants,
                              not the density you asked for
      mass_g                : from the measured density
      unsupported_frac      : fraction of surface area on downward-facing faces
                              steeper than the overhang limit. 0.000 = support-free.
      wall_thickness_mm     : half-isovalue estimate, distance-field topologies only
    """
    ext = mesh.bounds[1] - mesh.bounds[0]
    vol_box = float(tile_volume_mm3 if tile_volume_mm3 is not None
                    else ext[0] * ext[1] * ext[2])
    rho = float(mesh.volume) / vol_box if vol_box > 0 else float("nan")

    # A face needs support when it faces downward AND its plane is shallower than
    # the overhang limit off horizontal. Plane angle off horizontal = acos(|nz|),
    # so a vertical wall (nz = 0) is 90 deg -> always fine.
    nz = mesh.face_normals[:, 2]
    area = mesh.area_faces
    down = nz < 0
    shallow = np.degrees(np.arccos(np.clip(np.abs(nz), 0, 1))) < overhang_limit_deg
    # The tile's own bottom face is downward and flat, but it sits ON the build
    # plate -- counting it would report every prismatic (support-free) tile as
    # needing support. Exclude faces sitting within a hair of the lowest z.
    zmin = float(mesh.bounds[0][2])
    on_plate = mesh.triangles_center[:, 2] < zmin + 0.2      # mm, under one voxel
    unsupported = float(area[down & shallow & ~on_plate].sum() / max(area.sum(), 1e-9))

    wall = mesh.metadata.get("wall_thickness_mm")
    counts = mesh.metadata.get("cell_counts")
    n_across = min(counts[:2]) if isinstance(counts, (tuple, list)) and len(counts) >= 2 \
        else None
    out = {"watertight": bool(mesh.is_watertight),
           "rel_density_measured": rho,
           "mass_g": rho * vol_box * TPU_DENSITY_G_MM3,
           "unsupported_frac": unsupported,
           "wall_thickness_mm": wall,
           "cells_across": n_across,
           "n_bodies": int(mesh.body_count)}

    if verbose:
        wt = "watertight" if out["watertight"] else "NOT watertight - slicer will error"
        print(f"    Mesh: {mesh.volume:.0f} mm3, {wt}")
        print(f"    As-built density: {rho:.3f}  |  mass ~{out['mass_g']:.1f} g")
        if unsupported < 1e-4:
            print(f"    Overhangs: none past {overhang_limit_deg:.0f} deg "
                  f"- SUPPORT-FREE")
        else:
            print(f"    Overhangs: {unsupported*100:.1f}% of surface area needs "
                  f"support past {overhang_limit_deg:.0f} deg")
        if wall is not None:
            flag = "OK for 0.4mm nozzle" if wall >= min_wall_mm else \
                   f"BELOW {min_wall_mm}mm min - webbing/underextrusion risk"
            print(f"    Measured wall thickness: {wall:.2f} mm - {flag}")
        if n_across is not None and n_across < 4:
            print(f"    NOTE: only {n_across} cells across the 50.8mm span - coarse; "
                  f"few-cell coupons scatter more in crush. Reduce cell size if the "
                  f"wall stays >= {min_wall_mm}mm.")
        if out["n_bodies"] > 1:
            print(f"    WARNING: {out['n_bodies']} disconnected bodies - part will "
                  f"not hold together. For spinodoid use form='sheet'.")
    return out


def wall_thickness_for(topology, cell_size, relative_density, res=56, **kw):
    """As-built wall thickness (mm) of a wall-network topology at a given cell size
    and target density. MEASURED from a real density-corrected build, so it matches
    what actually comes out of build_tile -- not the raw quantile, which reads a bit
    thick because the density correction later trims the isovalue.

    A wall lattice has a fixed wall length per unit area L ~ 1/cell, and density = L*t,
    so a small cell forces a thin wall -- and TPU on a 0.4 mm nozzle cannot do thin.
    This is the constraint check_requirements' generic t = (rho/1.5)*cell estimate
    hides: at 8 mm re-entrant it predicts 1.33 mm where the truth is ~0.6 mm.
    """
    spec = FIELD_TOPOLOGIES[topology]
    if not spec["distance"]:
        raise ValueError(f"'{topology}' is not a wall/distance-field topology")
    mesh, _ = build_tile(topology, cell_size, relative_density, res=res,
                         drop_islands=False, **kw)
    return float(mesh.metadata.get("wall_thickness_mm", 0.0))


def suggest_cell_size(topology, relative_density, min_wall_mm=MIN_WALL_MM,
                      lo=4.0, hi=32.0, res=56, max_iter=8, **kw):
    """Smallest cell size whose AS-BUILT wall clears `min_wall_mm` at this density.

    Returns (cell_mm, wall_mm). Smaller cells are better for a helmet liner -- more
    cells through the thickness means a smoother crush plateau and less
    single-cell-buckling scatter -- so this bisects for the SMALLEST printable one.
    Bounded iterations because each step is a real (coarse) build.
    """
    def w(c):
        return wall_thickness_for(topology, c, relative_density, res=res, **kw)
    whi = w(hi)
    if whi < min_wall_mm:
        return hi, whi                       # even the coarsest cell cannot do it
    wlo = w(lo)
    if wlo >= min_wall_mm:
        return lo, wlo
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if w(mid) >= min_wall_mm:
            hi = mid
        else:
            lo = mid
    return hi, w(hi)


# ── Mesh repair ────────────────────────────────────────────────────────────────

def _repair(mesh):
    """Fix common marching-cubes mesh issues that cause slicer errors."""
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, ::-1]
        mesh.fix_normals()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    # Remove degenerate faces (API varies by trimesh version)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    return mesh


def save_tile(mesh, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.stl")
    mesh.export(path)
    return path


# ── Analytical checks ──────────────────────────────────────────────────────────

def check_requirements(name, mesh, relative_density, cell_size, source="MANUAL"):
    tag = "[ML-OPTIMIZED]" if source == "ML" else "[STARTING POINT - not yet optimized]"
    print(f"  {name}  {tag}")

    vol = mesh.volume
    watertight = mesh.is_watertight
    status = "watertight - OK for slicing" if watertight else "NOT watertight - slicer may error"
    if vol <= 0:
        print(f"    MESH ERROR: volume={vol:.0f}mm3. Do not print.")
        return
    print(f"    Mesh: {vol:.0f} mm3, {status}")

    mass = relative_density * (TILE_X * TILE_Y * TILE_Z) * TPU_DENSITY_G_MM3
    print(f"    Est. mass: ~{mass:.1f} g")

    sigma_peak = TARGET_FORCE_N / TILE_AREA_MM2
    E_star = 0.3 * TPU_Es_MPA * (relative_density ** 1.5)
    eps_d = max(1.0 - 1.4 * relative_density, 0.1)
    eps_linear = sigma_peak / E_star
    regime = "hyperelastic regime (expected for TPU, confirm spring-back at VT)" \
             if eps_linear > eps_d else "within estimated elastic range"
    print(f"    At 7100N: stress={sigma_peak:.3f} MPa, E*={E_star:.2f} MPa, regime: {regime}")

    t_wall = (relative_density / 1.5) * cell_size
    if t_wall < MIN_WALL_MM:
        print(f"    PRINTABILITY WARNING: est. wall {t_wall:.1f}mm < {MIN_WALL_MM}mm min - webbing risk")
    else:
        print(f"    Printability: est. wall {t_wall:.1f}mm - OK for 0.4mm nozzle")
    print()


# ── Generate from ML recommendations ──────────────────────────────────────────

def generate_from_recommendation(rec_id, topology, cell_size, relative_density, **kw):
    """
    Called by surrogate.py to auto-generate an STL for an ML-recommended design.
    rec_id: iteration label, e.g. 'R2_01'
    Extra kwargs (seed=, theta_deg=, mode=, irregularity=, ...) pass to the field, so
    a stochastic recommendation is reproducible from its logged parameters.
    Returns: (mesh, path, expected_mass)
    """
    if topology == "bouligand":
        mesh, expected_mass = generate_bouligand_tile(
            cell_size=cell_size, relative_density=relative_density, **kw)
    else:
        mesh, expected_mass = build_tile(topology, cell_size, relative_density, **kw)

    name = f"{rec_id}_{topology}_rho{relative_density:.2f}_cell{cell_size:.1f}mm"
    if kw.get("seed") is not None:
        name += f"_s{kw['seed']}"
    path = save_tile(mesh, name)
    check_requirements(name, mesh, relative_density, cell_size, source="ML")
    printability_report(mesh, tile_volume_mm3=TILE_X * TILE_Y * TILE_Z)
    return mesh, path, expected_mass


# ── Main: generate initial 5 designs ──────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING INITIAL TILES  [STARTING POINTS - not ML-optimized]")
    print("These cover the design space for round 1 VT testing.")
    print("ML optimization begins after you enter VT results.")
    print("=" * 60)
    print()

    designs = [
        # (name,                  fn,        cell, density, res)
        ("D1_gyroid_025_8mm",   gyroid,     8.0, 0.25, RESOLUTION),
        ("D2_gyroid_035_6mm",   gyroid,     6.0, 0.35, RESOLUTION),
        ("D4_schwartz_025_8mm", schwartz_p, 8.0, 0.25, RESOLUTION),
    ]

    for name, fn, cell, density, res in designs:
        print(f"  Generating {name}...")
        mesh, _ = generate_tile(fn, cell, density, res=res)
        path = save_tile(mesh, name)
        print(f"    Vertices: {len(mesh.vertices):,}  |  Saved: {path}")
        check_requirements(name, mesh, density, cell, source="MANUAL")

    print("  Generating D5_graded_gyroid_018to035...")
    mesh_d5, mass_d5 = generate_graded_tile()
    path_d5 = save_tile(mesh_d5, "D5_graded_gyroid_018to035")
    print(f"    Vertices: {len(mesh_d5.vertices):,}  |  Saved: {path_d5}")
    print(f"    Est. mass: ~{mass_d5:.1f} g  |  [STARTING POINT]")
    print()

    print("  Generating D3_rotating_squares_auxetic...")
    mesh_d3, mass_d3, rho_d3 = generate_rotating_squares()
    path_d3 = save_tile(mesh_d3, "D3_rotating_squares_auxetic")
    print(f"    Vertices: {len(mesh_d3.vertices):,}  |  Saved: {path_d3}")
    print(f"    Actual density: {rho_d3:.3f}  |  Est. mass: ~{mass_d3:.1f} g  |  [STARTING POINT]")
    print()

    # ── Auxetic (re-entrant) + biologically-inspired designs ──────────────────
    # All support-free: the fields are prismatic, so every wall is vertical.
    # D6/D7 are the auxetic family; D8-D11 are the bio-inspired set. Held at the
    # same rel_density 0.25 as D1/D4 so architecture is the only variable -- the
    # whole point of the geometry-only study.
    print("=" * 60)
    print("AUXETIC (RE-ENTRANT) + BIO-INSPIRED DESIGNS  [support-free]")
    print("=" * 60)
    print()

    # Cell sizes come from suggest_cell_size(): the smallest cell whose AS-BUILT wall
    # clears the 1.2 mm printable minimum. The field-based ones (voronoi/spinodoid)
    # have thick enough features at 8 mm and stay at rho=0.25 like D1/D4.
    #
    # The RE-ENTRANT auxetics sit at rho=0.35, not 0.25: a wall lattice this open
    # can't have both printable walls and enough cells across a 2 in tile at 0.25
    # (it needed a ~20 mm cell -> only 2 cells, which barely shows the negative-
    # Poisson effect). At 0.35 the walls thicken so a smaller cell prints, giving
    # 3-4 cells across. Density is an ML knob, so one family sitting higher is fine.
    bio = [
        # (name,                           topology,     cell, rho,  kwargs)
        ("D6_re_entrant_035_11mm",         "re_entrant", 11.3, 0.35, {}),
        ("D7_re_entrant_deep_035_17mm",    "re_entrant", 17.0, 0.35, {"theta_deg": -45.0}),
        ("D8_honeycomb_025_12mm",          "honeycomb",  11.3, 0.25, {}),
        ("D9_elytra_025_14mm",             "elytra",     14.6, 0.25, {}),
        ("D10_voronoi_bone_025_8mm",       "voronoi",     8.0, 0.25, {"seed": 1,
                                                                     "irregularity": 0.6}),
        ("D11_spinodoid_025_8mm",          "spinodoid",   8.0, 0.25, {"seed": 1,
                                                                     "mode": "lamellar"}),
    ]
    for name, topo, cell, rho, kw in bio:
        print(f"  Generating {name}...")
        mesh, _ = build_tile(topo, cell, rho, **kw)
        path = save_tile(mesh, name)
        print(f"    Vertices: {len(mesh.vertices):,}  |  Saved: {path}")
        printability_report(mesh, tile_volume_mm3=TILE_X * TILE_Y * TILE_Z)
        print()

    print("  Generating D12_bouligand_honeycomb_025_12mm...")
    mesh_b, mass_b = generate_bouligand_tile(base="honeycomb", cell_size=12.0,
                                             relative_density=0.25,
                                             n_plies=6, pitch_deg=60.0)
    path_b = save_tile(mesh_b, "D12_bouligand_honeycomb_025_12mm")
    print(f"    Vertices: {len(mesh_b.vertices):,}  |  Saved: {path_b}")
    printability_report(mesh_b, tile_volume_mm3=TILE_X * TILE_Y * TILE_Z)
    print()

    print("=" * 60)
    print("NEXT STEP: slice in Creality Print and verify no red regions.")
    print("Then print, QC (prints_log/print_checklist.md), and go to VT.")
    print("After VT: enter results in vt_results/results.csv and run")
    print("  python scripts/surrogate.py")
    print("to get ML-optimized design recommendations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
