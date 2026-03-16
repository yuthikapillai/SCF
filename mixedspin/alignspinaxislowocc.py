import numpy as np
import colorsys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

# ── PySCF Imports ────────────────────────────────────────────────────────────
from pyscf import gto
from pyscf.tools import cubegen

# ──GHF machinery ────────────────────────────────────────────────────────
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.ghf import GHF
from quantel.opt.lbfgs import LBFGS

# ══════════════════════════════════════════════════════════════════════════════
# 1. VISUALIZATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def phase_to_rgb(theta):
    hue = (theta % (2 * np.pi)) / (2 * np.pi)
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue.ravel()])
    return rgb.reshape(*theta.shape, 3)

def build_pyscf_mol(mol_obj):
    if hasattr(mol_obj, "mol"): return mol_obj.mol
    m = gto.Mole()
    m.atom, m.basis = mol_obj.atom, mol_obj.basis
    m.unit = getattr(mol_obj, "unit", "angstrom")
    m.spin, m.charge = getattr(mol_obj, "spin", 1), getattr(mol_obj, "charge", 0)
    m.build(); return m

def atom_traces(pyscf_mol):
    """Generates large white spheres for all atoms."""
    traces = []
    coords = pyscf_mol.atom_coords() 
    for i, sym in enumerate(pyscf_mol.elements):
        traces.append(go.Scatter3d(
            x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
            mode="markers",
            marker=dict(size=12, color="white", opacity=1.0, 
                        line=dict(width=1, color="black")),
            name=f"H{i}",
            showlegend=False
        ))
    return traces

def evaluate_complex_orbital_on_grid(pyscf_mol, complex_coeff, nx=150, ny=150, nz=150, margin=5.0):
    cube = cubegen.Cube(pyscf_mol, nx=nx, ny=ny, nz=nz, margin=margin)
    coords = cube.get_coords() 
    psi_grid = (pyscf_mol.eval_gto("GTOval_sph", coords) @ complex_coeff).reshape(nx, ny, nz)
    origin = getattr(cube, 'boxorig', getattr(cube, 'boxorigin', None))
    xs = np.linspace(origin[0], origin[0] + cube.box[0,0], nx)
    ys = np.linspace(origin[1], origin[1] + cube.box[1,1], ny)
    zs = np.linspace(origin[2], origin[2] + cube.box[2,2], nz)
    return xs, ys, zs, psi_grid

def make_orbital_mesh(xs, ys, zs, psi_grid, isovalue_fraction=0.08, name="ψ"):
    norm = np.abs(psi_grid)
    flat = np.sort(norm.ravel())[::-1]
    level = float(flat[max(np.searchsorted(np.cumsum(flat), isovalue_fraction * np.sum(flat)) - 1, 0)])
    verts, faces, _, _ = marching_cubes(norm, level=level)
    dx, dy, dz = (xs[-1]-xs[0])/(len(xs)-1), (ys[-1]-ys[0])/(len(ys)-1), (zs[-1]-zs[0])/(len(zs)-1)
    verts_real = np.column_stack([xs[0]+verts[:,0]*dx, ys[0]+verts[:,1]*dy, zs[0]+verts[:,2]*dz])
    interp = RegularGridInterpolator((xs, ys, zs), np.angle(psi_grid), method="linear", bounds_error=False, fill_value=0.0)
    v_colors = ["rgb({},{},{})".format(*(c*255).astype(int)) for c in phase_to_rgb(interp(verts_real))]
    
    return go.Mesh3d(x=verts_real[:,0], y=verts_real[:,1], z=verts_real[:,2],
                     i=faces[:,0], j=faces[:,1], k=faces[:,2],
                     vertexcolor=v_colors, opacity=1.0, name=name)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MAIN CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mol = PySCFMolecule([["H", 2,0,0],["H",0,2,0],["H",0,0,2]], "sto-3g", "angstrom", spin=1)
    ints = PySCFIntegrals(mol); S = ints.overlap_matrix()
    wfn = GHF(ints); wfn.initialise(np.random.rand(wfn.nmo, wfn.nmo))
    LBFGS().run(wfn); wfn.canonicalize()

    # SU(2) Rotation Logic
    nao = wfn.nmo // 2
    ci0 = wfn.mo_coeff[:, 0]
    pi0 = np.outer(ci0, ci0)
    sx = np.trace(0.5 * (pi0[:nao, nao:] + pi0[nao:, :nao]) @ S)
    sz = np.trace(0.5 * (pi0[:nao, :nao] - pi0[nao:, nao:]) @ S)
    s_vec = np.array([sx, 0, sz])
    
    n_vec = s_vec / np.linalg.norm(s_vec)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(n_vec, z_axis)
    axis_norm = np.linalg.norm(rot_axis)
    rot_axis = rot_axis / axis_norm if axis_norm > 1e-8 else np.array([0, 0, 1])
    theta = np.arccos(np.clip(np.dot(n_vec, z_axis), -1.0, 1.0))
    c, s = np.cos(theta/2), np.sin(theta/2)
    ux, uy, uz = rot_axis
    U = np.array([[c - 1j*uz*s, -(1j*ux + uy)*s], [-(1j*ux - uy)*s, c + 1j*uz*s]], dtype=complex)

    phi0_rot = U @ np.vstack([ci0[:nao], ci0[nao:]])
       # 1. Reconstruct the full rotated GHF coefficient vector (nao * 2)
       # alpha is phi0_rot[0], beta is phi0_rot[1]
    ci0_rot = np.hstack([phi0_rot[0, :], phi0_rot[1, :]])

# 2. Construct the Density Matrix (P = c * c_dagger) for this MO
# We use .conj() in case the rotation introduced complex phases
    pi0_rot = np.outer(ci0_rot, ci0_rot.conj())

# 3. Extract the spin blocks for Sx and Sz
# Sx = 0.5 * (P_ab + P_ba)
# Sz = 0.5 * (P_aa - P_bb)
    X_rot = 0.5 * (pi0_rot[:nao, nao:] + pi0_rot[nao:, :nao])
    Z_rot = 0.5 * (pi0_rot[:nao, :nao] - pi0_rot[nao:, nao:])

# 4. Trace with the Overlap matrix S to get the integrated spin expectation
    sx_new = np.trace(X_rot @ S).real
    sz_new = np.trace(Z_rot @ S).real
    sy_new = 0 # Sy is usually zero in these GHF implementations unless complex basis are used

    s_vec_rot = np.array([sx_new, sy_new, sz_new])

    print("\n--- Spin Vector Comparison ---")
    print(f"Original s_vec: ({s_vec[0]:.6f}, {s_vec[1]:.6f}, {s_vec[2]:.6f})")
    print(f"Rotated  s_vec: ({s_vec_rot[0]:.6f}, {s_vec_rot[1]:.6f}, {s_vec_rot[2]:.6f})")
    print(f"Total Magnitude: {np.linalg.norm(s_vec_rot):.6f}")
    complex_ci_rot = phi0_rot[0, :] + 1j * phi0_rot[1, :]
    print(phi0_rot)
    # Rendering Setup
    pyscf_mol = build_pyscf_mol(mol)
    atoms = atom_traces(pyscf_mol)
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Original MO 0", "Rotated MO 0"))

    # Left Plot
    xs, ys, zs, psi0_orig = evaluate_complex_orbital_on_grid(pyscf_mol, ci0[:nao] + 1j*ci0[nao:])
    fig.add_trace(make_orbital_mesh(xs, ys, zs, psi0_orig), row=1, col=1)
    for t in atoms: fig.add_trace(t, row=1, col=1)

    # Right Plot
    xs, ys, zs, psi0_rot = evaluate_complex_orbital_on_grid(pyscf_mol, complex_ci_rot)
    fig.add_trace(make_orbital_mesh(xs, ys, zs, psi0_rot), row=1, col=2)
    for t in atoms: fig.add_trace(t, row=1, col=2)

   
    # ── CLEAN LAYOUT (No axes, No background) ────────────────────────────────
    clean_scene = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="black" # Solid black background
    )

    fig.update_layout(
        title="H3 GHF MO 0 z axis spin Alignment",
        paper_bgcolor="black",
        font=dict(color="white"),
        scene=clean_scene,
        scene2=clean_scene,
        margin=dict(l=0, r=0, b=0, t=50)
    )
# Save the final comparison as a high-res PNG
fig.write_image("ghf_spin_alignment.png", width=1600, height=800, scale=2)
print("Image saved as 'ghf_spin_alignment.png'")

fig.show()
