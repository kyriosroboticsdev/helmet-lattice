import sys, re, io, contextlib
import numpy as np
sys.path.insert(0, r"<scratchpad>")
with contextlib.redirect_stdout(io.StringIO()):
    import cube_test as C

C10, C01, D1 = 7.895, -1.519, 0.0435
# Stability (incompressible MR): uniaxial nominal stress P = 2(lam - lam^-2)(C10 + C01/lam),
# equibiaxial P = 2(lam - lam^-5)(C10 + C01*lam^2). Instability where dP/dlam <= 0.
lam = np.linspace(0.3, 3.0, 5401)
for name, P in [("uniaxial", 2*(lam - lam**-2)*(C10 + C01/lam)),
                ("equibiaxial", 2*(lam - lam**-5)*(C10 + C01*lam**2))]:
    bad = lam[np.gradient(P, lam) <= 0]
    print(f"MR {name}: unstable stretches -> "
          + ("none in 0.3..3.0" if bad.size == 0 else f"{bad.min():.3f}..{bad.max():.3f}"))
mu0 = 2*(C10 + C01); K = 2/D1
print(f"MR mu0={mu0:.2f} MPa, K={K:.0f} MPa, nu={(3*K-2*mu0)/(2*(3*K+mu0)):.3f}")

card = r"\*HYPERELASTIC, OGDEN, N=3\n[^\n]*\n[^\n]*\n"
c, n, t = C.cube_mesh()
C.run("cube Mooney-Rivlin C10=7.895 C01=-1.519 D1=0.0435", c, n, t,
      lambda s: re.sub(card, f"*HYPERELASTIC, MOONEY-RIVLIN\n{C10}, {C01}, {D1}\n", s))
C.run("cube Neo-Hooke C10=5.005 D1=0.0435", c, n, t,
      lambda s: re.sub(card, f"*HYPERELASTIC, NEO HOOKE\n5.005, {D1}\n", s))
