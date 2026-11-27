# Uniaxial tension — base TPU, 2026-07-07

Five printed ASTM D412 Die C dogbones (plate `prints_log/gcode/CK1MAX_PLATE_dogbone_x5_t2mm.gcode`),
pulled at 500 mm/min, strain from a 1-inch extensometer. Raw machine export, one CSV per specimen
(header block with specimen width/thickness/gauge, then Time s / Extensometer mm / Load N / Machine Extension mm).

Summary (from the 2026-10-21 research brief): break at 385–454% strain and 17.7–20.0 MPa;
stress at 100% strain 8.75–9.11 MPa (4% spread), so the printing is repeatable.

This is one of the three deformation modes needed for the hyperelastic fit
(`scripts/material/fit_hyperelastic.py`); planar shear and equibiaxial are still to run.
Note the fit's gate wants matched-rate lab data for all modes.
