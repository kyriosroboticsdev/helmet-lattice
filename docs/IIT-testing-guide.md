# IIT Testing Guide — TPU Material Characterization (and lattice tiles)

**Purpose of the trip:** measure how your printed TPU behaves under load, in enough
different ways that you can fit a *material model* to it. That fitted model is what
lets the FEA/SimScale simulations of the helmet lattice actually be trusted. In
other words: these bench tests are the ground truth that calibrates everything the
ML pipeline predicts.

This doc assumes you're not steeped in the mechanics, so each test explains **what
the machine does**, **what you record**, and **why it matters** — plus the
engineering gotchas that ruin a specimen if you skip them.

---

## 0. The big idea (read this first — 3 minutes)

TPU is a **hyperelastic** material: rubber-like, highly stretchy, and it does **not**
follow a straight-line stress/strain relationship the way steel or aluminum do at
small strains. A single number like "Young's modulus" doesn't describe it. Instead
the material is described by a *strain-energy function* (Ogden, Mooney-Rivlin, etc.),
and fitting that function's coefficients requires seeing the material deform in
**several independent ways**.

That's why you printed multiple specimen shapes. Each one forces the material into a
different **deformation mode**:

| Mode | Specimen you printed | What it stretches |
|------|----------------------|-------------------|
| **Uniaxial tension** | ASTM D412 Die C dogbone | pulled in one direction |
| **Planar / pure shear** | wide-short strip | stretched but not allowed to narrow |
| **Equibiaxial** | disk-bulge, compression puck, *or* cruciform | stretched equally in two directions |

The material-fitting software (SimScale's Ogden fit) needs **all three modes** to
produce coefficients that don't blow up when the real lattice deforms in ways no
single test covered. One or two modes = an overfit material model that lies.

TPU is also **nearly incompressible** (it changes shape, not volume). A useful
consequence you'll use below: *equibiaxial tension is mechanically equivalent to
uniaxial compression* for an incompressible rubber — which is why a simple
lubricated compression puck can stand in for the hard-to-run biaxial test.

**Two more things that matter for every test:**
- **Print profile must be identical across specimens.** Same nozzle temp, same
  infill (these are *solid*, 100%), same orientation. If the specimens don't share a
  print history, you're characterizing three different materials, not three modes of
  one. This is why the specimens are all thin/flat and printed in the same profile.
- **These are SOLID base TPU, not lattices.** You're measuring the raw plastic here.
  The lattice geometry gets its own tests (Section 5) and its behavior is *predicted*
  from this material model.

---

## 1. Test 1 — Uniaxial Tension (the dogbone)

**Specimen:** `ASTM_D412_DieC_dogbone_t3mm.stl` — the bone/dumbbell shape.

**Geometry that matters:**
- Overall length **115 mm**, grip ends **25 mm** wide.
- **Narrow gauge section: 6.0 mm wide × 33 mm long**, thickness **3 mm** (default).
- Gauge cross-section = 6 × 3 = **18 mm²**. This is the number that sets your force:
  at ~10 MPa engineering stress that's only ~180 N, so a small load cell is fine.
- The flared shoulders (14 mm + 25 mm tangent radii) exist so the specimen breaks in
  the **middle**, not at the grips. A break at the grip = void the test.

**The machine:** a **UTM** (Universal Testing Machine — a screw-driven frame with two
grips and a load cell). The lower grip is fixed; the upper grip pulls up at a constant
speed. It records **force** (from the load cell) and **crosshead displacement** (how
far it pulled).

**How to run it:**
1. Clamp the two wide grip tabs, leaving the 33 mm gauge free between the jaws.
2. Pull at **500 mm/min** (the ASTM D412 rate).
3. Ideally track gauge strain with an **extensometer** or video/dot-tracking, not
   just crosshead travel — crosshead motion includes grip slip and machine compliance
   and will overstate the strain.
4. Pull to failure (or to the max strain you care about).

**What you record:** force vs. displacement (or force vs. strain). You'll convert:
- **Engineering stress** = force ÷ 18 mm² (original area).
- **Engineering strain** = extension ÷ gauge length (33 mm).

**Why it matters:** this is the backbone curve of the material model — the most
intuitive mode and the one the other two are anchored against.

**Gotchas:**
- **Grip slip** is the #1 problem with TPU (it's slippery). Use serrated/rubber-faced
  grips; watch for the specimen creeping out of the jaws.
- Mark two gauge lines and photograph, or the strain data is guesswork.
- Print flaws (a stray gap, a weak layer line) concentrate stress — a specimen that
  necks or tears off-center is suspect. That's why you print **5 copies** and keep
  the consistent ones.

---

## 2. Test 2 — Planar / "Pure Shear" (the wide strip)

**Specimen:** `Planar_pureshear_100x10gauge_t2mm.stl` — a wide, short rectangle.

**Geometry that matters:**
- **100 mm wide**, with a **10 mm-tall free gauge** between the clamps → a **10:1
  aspect ratio**. That wide-and-short shape is the whole trick.
- **15 mm clamp allowance** above and below the gauge (so total height ~40 mm). That
  extra material is what the grips bite; it's *not* part of the measurement zone.
- Thickness **2 mm** uniform.

**The concept (why it's called "pure shear"):** when you clamp the **full width** and
pull vertically, the material *wants* to get narrower (like the dogbone necks) — but
the clamped full-width edges physically prevent it from contracting sideways. Held
that way, the little 10 mm gauge is forced into a state that's mechanically equivalent
to **pure shear**, a distinct deformation mode from tension. No fancy shear rig
needed; the geometry creates the shear state.

**The machine:** same UTM as Test 1, but with **wide grips** that clamp the entire
100 mm top and bottom edges (full-width clamping is essential — partial grips ruin the
constraint that makes it "pure shear").

**How to run it:**
1. Clamp the full width top and bottom, leaving only the 10 mm gauge free.
2. Pull slowly and steadily (a similar low rate to Test 1 is fine; keep it consistent).
3. Record force vs. displacement across that 10 mm gauge.

**What you record:** force vs. gauge extension. Stress = force ÷ (100 mm × 2 mm cross
section); strain = extension ÷ 10 mm.

**Why it matters:** it fills in the middle of the deformation space between uniaxial
tension and equibiaxial — the mode a real lattice cell wall sees when it bends and
shears. Skipping it is a common reason Ogden fits misbehave.

**Gotchas:**
- The **10:1 aspect ratio and full-width clamping are non-negotiable** — if the grips
  only catch part of the width, the edges pull in and you're measuring something
  between tension and shear, not pure shear.
- Keep the gauge short; if it slips and the free height grows, the constraint weakens.

---

## 3. Test 3 — Equibiaxial (three possible routes)

Equibiaxial = the material is stretched **equally in two perpendicular directions at
once** (imagine the top of an inflating balloon). It's the hardest mode to produce,
so you printed options. **Pick the route that matches the machine IIT actually has.**

### Route A — Compression puck (easiest; needs only a normal UTM) ✅ recommended default
**Specimen:** `Compression_puck_D28.6_h12.5mm.stl` — a solid cylinder, **Ø28.6 mm ×
12.5 mm tall** (ASTM D575-style).

**The trick:** because TPU is nearly incompressible, **squashing it uniaxially is
mathematically the same deformation as stretching it equibiaxially.** So you get your
equibiaxial data from a simple compression test — no special rig.

**How to run it:**
1. Put the puck between two flat compression platens on the UTM.
2. **Lubricate both flat faces** (grease/PTFE). This is critical: friction between
   puck and platen stops the material sliding outward, makes it barrel, and
   contaminates the data. Lubrication keeps the compression uniform (uniaxial).
3. Compress at a slow, steady rate; record force vs. platen displacement.

**Record:** force vs. compressive displacement. Stress = force ÷ puck area
(π·14.3² ≈ 642 mm²); strain = displacement ÷ 12.5 mm.

**Gotcha:** re-lubricate between runs; if you see the puck **barrel** (bulge in the
middle), friction is winning — stop and add lubricant.

### Route B — Disk-bulge / bubble inflation (a true membrane biaxial test)
**Specimen:** `Equibiaxial_bulge_disk_D120_t2mm.stl` — a flat **120 mm** disk, **2 mm**
thick.

**The trick:** clamp the disk's outer ring over a circular aperture and inflate it
with air/fluid from below. The center domes up and the apex is in genuine equibiaxial
tension.

**How to run it:** clamp the outer ring (the disk is deliberately larger than the rig
aperture so there's clamping material), pressurize, and record **inflation pressure**
plus **dome apex height** (usually via a laser/camera). Those two convert to biaxial
stress/strain.

**Needs:** a bulge/inflation rig with pressure control and apex-height measurement —
not every lab has one. If IIT doesn't, use Route A.

### Route C — Cruciform (true equibiaxial, but needs a 4-arm biaxial machine)
**Specimen:** `Cruciform_biaxial_120mm_t2mm.stl` — the **plus/cross shape** you just
printed 3 of.

**Geometry:** **120 mm tip-to-tip each axis**, **30 × 30 mm central gauge**, **5 mm**
inner fillets, **5 slits per arm** (making 6 "fingers"), **2 mm** thick. The slits let
each arm stretch and spread independently so the center square sees **even, equal
biaxial strain** instead of the arms fighting each other.

**How to run it:** mounts in a **biaxial / 4-arm test machine** that pulls all four
arms outward simultaneously. Strain in the 30×30 center is tracked optically (DIC or
dot tracking). **This only works on a biaxial machine** — you cannot run it on a
normal single-axis UTM. If IIT has a planar-biaxial frame, this is the gold-standard
equibiaxial data; otherwise, Route A is the practical substitute.

> **Decision rule:** Normal UTM only → **Route A (compression puck)**. Bulge rig
> available → optionally add **Route B**. 4-arm biaxial machine available → **Route C
> (cruciform)** for the best data. You do **not** need all three; you need one good
> equibiaxial dataset.

---

## 4. Putting the three modes together (what happens after testing)

You'll leave IIT with three stress–strain curves: **uniaxial (Test 1)**, **planar
(Test 2)**, and **equibiaxial (Test 3, any route)**. Feed all three into SimScale's
(or Ansys/Abaqus's) **hyperelastic curve fitter**, choosing an **Ogden** model
(Mooney-Rivlin also works). The fitter finds the strain-energy coefficients that best
match all three curves *simultaneously*.

- Fitting to all three → a stable model that behaves under any deformation the lattice
  experiences.
- Fitting to one → coefficients that look fine in that mode and **diverge** elsewhere,
  making the FEA nonphysical.

That fitted material model is then assigned to the lattice geometry in simulation —
which is the whole reason for this trip.

---

## 5. Lattice tile tests (only if you're bringing the printed tiles)

If you also printed **lattice tiles** (`generate_tiles.py` — the ~**50.8 × 50.8 ×
38.1 mm**, i.e. 2 × 2 × 1.5 in prisms), these are tested *differently* from the
material specimens above. Here you're measuring how the **structure** behaves, not the
raw plastic.

**Quasi-static compression (UTM):**
- Compress the tile flat between platens at a slow, constant rate.
- Record force vs. displacement → you get the **plateau stress** (the roughly flat
  region where cells buckle and absorb energy) and the **densification strain** (where
  the crushed cells pack solid and force spikes up).
- The area under the curve is **energy absorbed** — the property that protects a head.
- Design target in the code is around **7100 N** peak force for a full tile; make sure
  the load cell range covers that.

**Impact / drop test (if the rig is available):**
- A weighted striker drops onto the tile; an accelerometer records the deceleration.
- Gives **peak acceleration (g)** and lets you compute injury metrics (**HIC**,
  linear; **BrIC**, rotational) — the same objectives the ML surrogate predicts.
- This is the real-world validation of the ML recommendations.

**Gotcha:** note the tile **orientation** relative to the print direction — lattices
are anisotropic, and compressing along vs. across the print layers gives different
numbers. Record which way is "up."

---

## 6. Bring / do checklist

**Bring:**
- [ ] 5× dogbones (Test 1), plus spares
- [ ] 5× planar strips (Test 2)
- [ ] Equibiaxial specimens for your chosen route (5× compression pucks is the safe
      default; cruciforms only if IIT has a biaxial machine)
- [ ] Lattice tiles (if doing Section 5)
- [ ] Calipers — **measure and record actual printed dimensions** of each gauge
      (printed TPU rarely matches nominal; the real cross-section sets your stress)
- [ ] Lubricant/grease (for compression pucks)
- [ ] Marker + ruler (gauge marks) and a phone/camera for strain tracking
- [ ] This doc + a blank data sheet

**For every test, log:**
- Specimen ID and which of the 5 copies it is
- **Measured** width/thickness/diameter (not nominal)
- Test speed (mm/min) and the machine/load-cell used
- Raw force–displacement data export (CSV if the machine allows)
- Anything weird: grip slip, off-center break, barreling, delamination

**Ask the IIT technician up front:**
1. What machine(s) are available — plain UTM, or a biaxial/4-arm frame? (decides your
   Test 3 route)
2. Load cell range (must cover ~180 N for dogbones and ~7100 N for lattice tiles).
3. Is an extensometer or DIC/video strain system available? (crosshead-only strain is
   much lower quality)
4. Can you export raw CSV, and at what sampling rate?

---

### One-line summary
You're running the **three hyperelastic modes** (tension, pure shear, equibiaxial) on
solid TPU coupons to fit a material model, and — if you brought them — **compression/
impact tests** on the lattice tiles to measure and validate energy absorption. Match
your equibiaxial method to the machine IIT has, measure the *real* printed dimensions,
and export the raw force–displacement curves.
