#!/usr/bin/env python3
"""iDMRG of the frustrated kagome Bose-Hubbard model on cylinders (TeNPy).

    H = t sum_<ij> (b^dag_i b_j + h.c.) + (U/2) sum_i n_i (n_i - 1)
        + V sum_<ij> (n_i - 1)(n_j - 1),      U = 1, t > 0 frustrated,

the convention of nlce_run, at unit filling, with at most Nmax bosons per
site, on an infinite cylinder of Ly unit cells around (TeNPy's Kagome
lattice, periodic along a2, MPS unit cell of Lx x Ly cells).

A chiral Mott insulator (CMI) breaks time reversal with loop currents while
the charge sector stays gapped.  Three runs per point:

    free        from the Mott product state, real, no bias
    staggered   a field h along the staggered current (up and down triangles
                circulating oppositely), ramped down to h = 0, each step
                continuing from the last
    uniform     the same for the uniform current (all triangles alike)

kappa/h along the ramp is the chiral susceptibility.  A chiral phase keeps
its current at h = 0, at an energy no higher than the free run's (which at
finite bond dimension can only represent the symmetric cat state).  Observables, chosen to tell the CMI
from its competitors:

    E/site, entanglement S
    J_b       current on each bond (ccw around its triangle): chirality
              per up/down triangle, i.e. staggered vs uniform pattern
    K_b       kinetic bond energy <b^dag b + h.c.>: trimerisation (up vs
              down) and nematic (orientation) bond order
    n_i, P3   densities (charge order) and the weight on n = Nmax
    xi_b      correlation length in the charge-1 sector: finite in a Mott
              insulator, diverging with chi in a superfluid
    xi_0      neutral-sector correlation length (any order or soft mode)
    C_kappa   up-triangle chirality correlations along the cylinder
    C_K, C_n  bond-energy and density correlations, for bond and charge
              order at any wavevector along the cylinder

Usage: kagome_dmrg.py --t 0.2,0.3 --v 0.25 --ly 2 --chi 400 --out DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.models.lattice import Kagome
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import BosonSite

# Nearest-neighbour pairs of TeNPy's Kagome lattice, (u1, u2, dx), with the
# sign of u1 -> u2 counterclockwise around its triangle and whether that
# triangle points up.  The unit cell (sites 0, 1, 2 at (0,0), (1,0),
# (1/2, sqrt3/2)) is an up triangle; the bonds between cells form the down
# triangles.
BONDS = [
    (0, 1, (0, 0), +1, True),
    (0, 2, (0, 0), -1, True),
    (1, 2, (0, 0), +1, True),
    (1, 0, (1, 0), -1, False),
    (2, 0, (0, 1), +1, False),
    (2, 1, (-1, 1), -1, False),
]

PATTERNS = {
    "staggered": lambda up: 1 if up else -1,
    "uniform": lambda up: 1,
}


class KagomeBoseHubbard(CouplingMPOModel):
    def init_sites(self, p):
        return BosonSite(Nmax=p.get("Nmax", 3), conserve="N", filling=1.0)

    def init_lattice(self, p):
        site = self.init_sites(p)
        return Kagome(p.get("Lx", 1), p.get("Ly", 2), [site] * 3, bc="periodic", bc_MPS="infinite")

    def init_terms(self, p):
        t, U, V = p.get("t", 0.2), p.get("U", 1.0), p.get("V", 0.0)
        h, pattern = p.get("h", 0.0), p.get("pattern", None)
        for u in range(3):
            self.add_onsite(U / 2, u, "NN")
            self.add_onsite(-U / 2, u, "N")
        for u1, u2, dx, ccw, up in BONDS:
            self.add_coupling(t, u1, "Bd", u2, "B", list(dx), plus_hc=True)
            self.add_coupling(V, u1, "dN", u2, "dN", list(dx))
            if h and pattern:
                # -h * sign * J, J_{u1->u2} = -i b^dag_u1 b_u2 + h.c. (ccw sign)
                w = -h * ccw * PATTERNS[pattern](up)
                self.add_coupling(-1j * w, u1, "Bd", u2, "B", list(dx), plus_hc=True)


def bond_sites(model):
    """MPS indices (i, j) of every bond in the MPS unit cell, with its BONDS entry."""
    out = []
    for k, (u1, u2, dx, ccw, up) in enumerate(BONDS):
        i, j, _, _ = model.lat.possible_couplings(u1, u2, list(dx))
        for a, b in zip(i, j):
            out.append((int(a), int(b), k))
    return out


def hop(psi, i, j):
    """<b^dag_i b_j> on an infinite MPS, any order of i, j."""
    if i < j:
        return complex(psi.expectation_value_term([("Bd", i), ("B", j)]))
    return complex(np.conj(psi.expectation_value_term([("Bd", j), ("B", i)])))


def measure(psi, model, ncorr):
    L = psi.L
    bonds = bond_sites(model)
    J, K = [], []
    for i, j, k in bonds:
        z = hop(psi, j, i)  # <b^dag_j b_i>
        J.append(-2 * z.imag * BONDS[k][3])  # current around the triangle, ccw
        K.append(2 * z.real)
    up = [x for x, (_, _, k) in zip(J, bonds) if BONDS[k][4]]
    dn = [x for x, (_, _, k) in zip(J, bonds) if not BONDS[k][4]]
    Kup = [x for x, (_, _, k) in zip(K, bonds) if BONDS[k][4]]
    Kdn = [x for x, (_, _, k) in zip(K, bonds) if not BONDS[k][4]]
    n = psi.expectation_value("N")
    p3 = []
    site = psi.sites[0]
    for i in range(L):
        rho = psi.get_rho_segment([i]).to_ndarray().reshape(site.dim, site.dim)
        p3.append(float(np.real(rho[-1, -1])))
    res = {
        "S": [float(s) for s in psi.entanglement_entropy()],
        "J_up": up, "J_down": dn, "K_up": Kup, "K_down": Kdn,
        "kappa_up": float(np.mean(up)), "kappa_down": float(np.mean(dn)),
        "n": [float(x) for x in np.real(n)], "P_Nmax": p3,
        "xi_b": float(psi.correlation_length(charge_sector=[1])),
        "xi_0": float(psi.correlation_length(charge_sector=[0])),
    }
    # Correlations along the cylinder, from the first up triangle (sites 0, 1, 2
    # of the first cell) to the same triangle r MPS unit cells further.
    tri = [(0, 1, +1), (1, 2, +1), (2, 0, +1)]

    def kappa_terms(off):
        return [(a + off, b + off, s) for a, b, s in tri]

    ck, cK, cn = [], [], []
    for r in range(1, ncorr + 1):
        off = r * L
        acc_k = acc_K = 0.0
        for a, b, s in kappa_terms(0):
            for c, d, s2 in kappa_terms(off):
                # J_{a->b} J_{c->d}, J_{a->b} = -i b^dag_a b_b + i b^dag_b b_a
                for (o1, o2, f1) in ((a, b, -1j), (b, a, 1j)):
                    for (o3, o4, f2) in ((c, d, -1j), (d, c, 1j)):
                        v = expect4(psi, o1, o2, o3, o4)
                        acc_k += np.real(f1 * f2 * v)
                        acc_K += np.real(v)
        ck.append(acc_k / 9)
        cK.append(acc_K / 9)
        cn.append(float(np.real(psi.expectation_value_term([("N", 0), ("N", off)])) - n[0] * n[0]))
    res["C_kappa"] = ck
    res["C_K"] = cK
    res["C_n"] = cn
    res["C_b"] = [abs(hop(psi, 0, r * L)) for r in range(1, ncorr + 1)]
    return res


def expect4(psi, a, b, c, d):
    """<b^dag_a b_b b^dag_c b_d> with a, b < c, d (different triangles)."""
    ops = sorted([(a, "Bd"), (b, "B")]) + sorted([(c, "Bd"), (d, "B")])
    return complex(psi.expectation_value_term([(op, i) for i, op in ops]))


def run(params, psi, chi, sweeps):
    model = KagomeBoseHubbard(params)
    if psi is None:
        L = model.lat.N_sites
        psi = MPS.from_product_state(model.lat.mps_sites(), [1] * L, bc="infinite",
                                     unit_cell_width=model.lat.mps_unit_cell_width)
    opts = {
        "mixer": True,
        "max_E_err": 1e-9,
        "max_S_err": 1e-6,
        "min_sweeps": sweeps[0],
        "max_sweeps": sweeps[1],
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "chi_list": {0: max(32, chi // 8), 4: max(64, chi // 4), 8: chi // 2, 12: chi},
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, model, opts)
    E, psi = eng.run()
    return float(E), psi, model


def report(t, v, name, h, m, ncorr):
    print("  t=%.3f V=%.3f %-9s h=%.3f E=%.8f S=%.3f kappa_up=%+.4f kappa_dn=%+.4f K_up=%.4f K_dn=%.4f "
          "xi_b=%.2f xi_0=%.2f P3=%.1e C_kappa(%d)=%.2e (%.0fs)" % (
              t, v, name, h, m["E"], max(m["S"]), m["kappa_up"], m["kappa_down"], np.mean(m["K_up"]),
              np.mean(m["K_down"]), m["xi_b"], m["xi_0"], max(m["P_Nmax"]), ncorr, m["C_kappa"][-1],
              m["seconds"]), flush=True)


def point(t, v, ly, lx, chi, nmax, ramp, ncorr):
    """The free run, then for each pattern a ramp of the pinning field down to
    zero, each step continuing from the last; E is always that of the model
    without the field."""
    base = {"t": t, "V": v, "U": 1.0, "Ly": ly, "Lx": lx, "Nmax": nmax}
    out = {"t": t, "V": v, "Ly": ly, "Lx": lx, "chi": chi, "Nmax": nmax, "ramp": ramp}
    t0 = time.time()
    _, psi, model = run(base, None, chi, (16, 40))
    m = measure(psi, model, ncorr)
    m["E"] = float(np.real(model.H_MPO.expectation_value(psi)))
    m["seconds"] = time.time() - t0
    out["free"] = m
    report(t, v, "free", 0.0, m, ncorr)
    model0 = model
    for name in ("staggered", "uniform"):
        steps = []
        psi = None
        for h in list(ramp) + [0.0]:
            t0 = time.time()
            p = {**base, "h": h, "pattern": name} if h else base
            _, psi, model = run(p, psi, chi, (8 if psi is not None else 12, 30))
            psi.canonical_form()
            m = measure(psi, model0, ncorr if h == 0 else 1)
            m["E"] = float(np.real(model0.H_MPO.expectation_value(psi)))
            m["h"] = h
            m["seconds"] = time.time() - t0
            steps.append(m)
            report(t, v, name, h, m, ncorr if h == 0 else 1)
        out[name] = steps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", required=True, help="comma-separated t/U")
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--ly", type=int, default=2)
    ap.add_argument("--lx", type=int, default=1)
    ap.add_argument("--chi", type=int, default=400)
    ap.add_argument("--nmax", type=int, default=3)
    ap.add_argument("--ramp", default="0.04,0.02,0.01,0.005", help="pinning fields of the biased runs, then 0")
    ap.add_argument("--ncorr", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in [float(x) for x in args.t.split(",")]:
        res = point(t, args.v, args.ly, args.lx, args.chi, args.nmax, [float(x) for x in args.ramp.split(",")],
                    args.ncorr)
        name = "t%.3f_v%.3f_ly%d_lx%d_chi%d.json" % (t, args.v, args.ly, args.lx, args.chi)
        (out / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
