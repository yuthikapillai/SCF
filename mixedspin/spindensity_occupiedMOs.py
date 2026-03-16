import numpy as np
import colorsys
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

# ── PySCF Imports ────────────────────────────────────────────────────────────
from pyscf import gto
from pyscf.tools import cubegen

# ── Your GHF machinery ────────────────────────────────────────────────────────
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.ghf import GHF
from quantel.opt.lbfgs import LBFGS

np.set_printoptions(linewidth=1000, precision=6, suppress=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Colour-wheel Helpers
# ══════════════════════════════════════════════════════════════════════════════

def phase_to_rgb(theta: np.ndarray) -> np.ndarray:
    """Map phase angles θ (radians) → RGB triples using the HSV wheel."""
    hue = (theta % (2 * np.pi)) / (2 * np.pi)
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue.ravel()])
    return rgb.reshape(*theta.shape, 3)

def rgb_to_plotly_colorscale(rgb_flat: np.ndarray):
    """Convert RGB float array to Plotly-compatible rgb strings."""
    rgb_uint8 = (rgb_flat * 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb_uint8]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Grid Evaluation & Molecule Logic
# ══════════════════════════════════════════════════════════════════════════════

def build_pyscf_mol(mol_obj) -> gto.Mole:
    if hasattr(mol_obj, "mol"):
        return mol_obj.mol
    m = gto.Mole()
    m.atom = mol_obj.atom
    m.basis = mol_obj.basis
    m.unit = getattr(mol_obj, "unit", "angstrom")
    m.spin = getattr(mol_obj, "spin", 1)
    m.charge = getattr(mol_obj, "charge", 0)
    m.verbose = 0
    m.build()
    return m

def evaluate_complex_orbital_on_grid(pyscf_mol, complex_coeff, nx=60, ny=60, nz=60, margin=4.0):
    """Evaluates MOs manually to avoid version-specific cubegen file errors."""
    cube = cubegen.Cube(pyscf_mol, nx=nx, ny=ny, nz=nz, margin=margin)
    coords = cube.get_coords() 
    
    # Contract AOs to get the complex wavefunction
    ao_values = pyscf_mol.eval_gto("GTOval_sph", coords)
    psi_flat = ao_values @ complex_coeff
    psi_grid = psi_flat.reshape(nx, ny, nz)
    
    # Version-safe attribute access for the box origin
    origin = getattr(cube, 'boxorig', getattr(cube, 'boxorigin', None))
    if origin is None:
        raise AttributeError("Could not find 'boxorig' or 'boxorigin' on Cube object.")

    # Generate axes
    xs = np.linspace(origin[0], origin[0] + cube.box[0,0], nx)
    ys = np.linspace(origin[1], origin[1] + cube.box[1,1], ny)
    zs = np.linspace(origin[2], origin[2] + cube.box[2,2], nz)
    
    return xs, ys, zs, psi_grid

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Mesh Creation & Atom Viz
# ══════════════════════════════════════════════════════════════════════════════

def make_orbital_mesh(xs, ys, zs, psi_grid, isovalue_fraction=0.08, name="ψ"):
    norm_grid = np.abs(psi_grid)
    phase_grid = np.angle(psi_grid)

    flat_sorted = np.sort(norm_grid.ravel())[::-1]
    cumsum = np.cumsum(flat_sorted)
    idx = np.searchsorted(cumsum, isovalue_fraction * cumsum[-1])
    level = max(float(flat_sorted[max(idx - 1, 0)]), norm_grid.max() * 0.02)

    verts, faces, _, _ = marching_cubes(norm_grid, level=level)

    dx, dy, dz = (xs[-1]-xs[0])/(len(xs)-1), (ys[-1]-ys[0])/(len(ys)-1), (zs[-1]-zs[0])/(len(zs)-1)
    verts_real = np.column_stack([xs[0] + verts[:, 0]*dx, ys[0] + verts[:, 1]*dy, zs[0] + verts[:, 2]*dz])

    phase_interp = RegularGridInterpolator((xs, ys, zs), phase_grid, 
                                           method="linear", bounds_error=False, fill_value=0.0)
    vertex_phases = phase_interp(verts_real)
    plotly_col = rgb_to_plotly_colorscale(phase_to_rgb(vertex_phases))

    return go.Mesh3d(
        x=verts_real[:, 0], y=verts_real[:, 1], z=verts_real[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=plotly_col, opacity=0.9,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3),
        name=name
    )

def atom_traces(pyscf_mol):
    traces = []
    colors = {"H": "#DDDDDD", "C": "#444444", "O": "#FF5555"}
    coords = pyscf_mol.atom_coords()
    for i, sym in enumerate(pyscf_mol.elements):
        traces.append(go.Scatter3d(
            x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
            mode="markers", marker=dict(size=8, color=colors.get(sym, "gray")),
            showlegend=False
        ))
    return traces

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main Visualisation & Export
# ══════════════════════════════════════════════════════════════════════════════

def visualise_ghf_orbitals(wfn, mol_obj, orbital_indices=None, nx=60, ny=60, nz=60, 
                           margin=4.0, isovalue_fraction=0.08, title="GHF MO Visualisation"):
    pyscf_mol = build_pyscf_mol(mol_obj)
    nao = wfn.nmo // 2
    if orbital_indices is None: orbital_indices = [0]
    n_orbs = len(orbital_indices)

    fig = make_subplots(rows=1, cols=n_orbs, specs=[[{"type": "scene"}] * n_orbs],
                        horizontal_spacing=0.02)

    for col_idx, mo_idx in enumerate(orbital_indices, start=1):
        print(f"  Processing MO {mo_idx}...")
        ci = wfn.mo_coeff[:, mo_idx]
        complex_ci = ci[:nao] + 1j * ci[nao:]

        xs, ys, zs, psi_grid = evaluate_complex_orbital_on_grid(pyscf_mol, complex_ci, nx, ny, nz, margin)
        mesh = make_orbital_mesh(xs, ys, zs, psi_grid, isovalue_fraction, name=f"MO {mo_idx}")
        
        for tr in [mesh] + atom_traces(pyscf_mol):
            fig.add_trace(tr, row=1, col=col_idx)

        scene_key = f"scene{col_idx}" if col_idx > 1 else "scene"
        fig.layout[scene_key].update(
            bgcolor="rgb(12,12,18)", aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
        )

    phase_legend = dict(text="<b>Phase θ</b><br>Red: 0<br>Yellow: π/2<br>Cyan: ±π<br>Purple: -π/2",
                        xref="paper", yref="paper", x=1.02, y=0.5, showarrow=False,
                        font=dict(color="white", size=10), bgcolor="rgba(255,255,255,0.5)")

    fig.update_layout(title=dict(text=title, x=0.5, font=dict(color="white")),
                      paper_bgcolor="rgb(0,0,0)", #margin=dict(r=150),
                      annotations=[phase_legend])

    # Save PNG
    save_path = "ghf_orbitals_final.png"
    print(f"  Saving high-resolution image to {save_path}...")
    try:
        fig.write_image(save_path, width=800*n_orbs, height=800, scale=2)
        print("  ✓ Image saved.")
    except Exception as e:
        print(f"  ! PNG export failed (requires kaleido): {e}")

    fig.show()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mol = PySCFMolecule([["H", 2.0, 0, 0], ["H", 0, 2.0, 0], ["H", 0, 0, 2.0]], 
                        "sto-3g", "angstrom", spin=1, charge=0)
    ints = PySCFIntegrals(mol)
    wfn = GHF(ints)
    wfn.initialise(np.random.rand(wfn.nmo, wfn.nmo))
    LBFGS().run(wfn)
    wfn.canonicalize()

    visualise_ghf_orbitals(wfn, mol, orbital_indices=[0, 1, 2], nx=150, ny=150, nz=150)