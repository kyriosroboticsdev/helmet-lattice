import sys, io, contextlib
sys.path.insert(0, r"<scratchpad>")
with contextlib.redirect_stdout(io.StringIO()):
    import cube_test as C
import numpy as np, dyn_lib as L
c, n, t = C.cube_mesh()
s = L.run("cube static 20%", c, n, t, 0.20, "static", 300)
d = L.run("cube dynamic 20%", c, n, t, 0.20, "dynamic", 300, T=0.05)
for e in (0.05, 0.10, 0.20):
    print(f"  {e*100:.0f}%: static {np.interp(e, s['strain'], s['stress_kpa']):.1f} kPa | dynamic {np.interp(e, d['strain'], d['stress_kpa']):.1f} kPa")
print(f"  peak kinetic energy / external work = {L.ke_ratio(d, c):.2e}")
