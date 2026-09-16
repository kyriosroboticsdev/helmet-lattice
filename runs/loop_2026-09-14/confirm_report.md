# Confirm report: 2x2x2 piece vs full tile

Both at a fixed 10% crush. Offset = (piece - full) / full.

| design | piece stress@10% kPa | full stress@10% kPa | offset | piece SEA@10% | full SEA@10% | offset | full status |
|---|---|---|---|---|---|---|---|
| it018_1 | 1047.9 | 1054.1 | -0.6% | 122.1 | 122.4 | -0.3% | ok |
| it020_1 | 825.5 | 842.7 | -2.0% | 122.8 | 125.8 | -2.4% | ok |
| it014_0 | 787.3 | 810.2 | -2.8% | 118.5 | 119.6 | -1.0% | ok |
| it020_0 | 796.9 | nan | +nan% | 121.3 | nan | +nan% | ok |
| it022_3 | 829.8 | 852.6 | -2.7% | 126.2 | 127.7 | -1.1% | ok |

Mean stress offset -2.0% (range -2.8% to -0.6%); mean SEA offset -1.2%; n=4.

Final screening hypervolume (fixed reference): 3.323e+05
