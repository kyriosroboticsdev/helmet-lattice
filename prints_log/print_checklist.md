# Print QC Checklist

Fill this out for every tile immediately after printing.

---

## For Each Tile

**Design ID:** _______________
**Print date:** _______________

### Dimensional Check (calipers)
| Dimension | Target  | Measured | Delta |
|-----------|---------|----------|-------|
| Length X  | 80.0 mm | _____ mm | _____ |
| Width Y   | 80.0 mm | _____ mm | _____ |
| Height Z  | 20.0 mm | _____ mm | _____ |

Acceptable tolerance: ± 0.5 mm. If outside this, reprint.

### Mass Check (kitchen scale, 0.1g resolution minimum)
| Value              | Result  |
|--------------------|---------|
| Measured mass (g)  | _____   |
| Expected mass (g)  | _____   (from generate_tiles.py output) |
| Delta %            | _____   |

Actual relative density = measured_mass_g / (80 × 80 × 20 × 0.00121)

If delta > 15% → adjust density_to_isovalue() in generate_tiles.py and reprint.

### Visual QC
- [ ] No visible layer delamination
- [ ] No collapsed struts or walls
- [ ] Minor stringing acceptable, large blobs are not
- [ ] Clean edges on base layer

### Manual Spring-Back Test
Compress firmly along Z axis, hold 3 seconds, release.

- [ ] Compresses smoothly (no cracking)
- [ ] Returns to original shape within 2 seconds
- [ ] No permanent deformation visible

Estimated stiffness (circle one): VERY SOFT / SOFT / MEDIUM / STIFF / VERY STIFF

### Notes
_____________________________________________________________
