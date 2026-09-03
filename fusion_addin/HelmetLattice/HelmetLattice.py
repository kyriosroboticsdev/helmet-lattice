"""
HelmetLattice — Fusion 360 add-in
=================================
Generate a TPU helmet-liner lattice tile and drop it straight into the active
Fusion design as a mesh body.

How it works
------------
Fusion's bundled Python has no numpy / scikit-image / trimesh, so this add-in is
just the UI. On "Generate" it shells out to the project's CLI...

    <PYTHON_EXE>  <REPO_DIR>/scripts/make_stl.py  --topology ... --out <temp.stl>

...using the *system* Python (which has the packages), then imports the resulting
STL as a mesh body. Edit the two CONFIG paths below if your install differs.

Topologies: gyroid / schwartz / diamond, graded, auxetic, alternating (Z-layers),
and `blend` — the GENERATIVE option that mixes the TPMS basis (gyroid/schwartz/
diamond) into a new in-between geometry. "Load latest proposal" pulls the
optimizer's recommended design from outputs/proposed_designs.csv.
"""

import adsk.core
import adsk.fusion
import ast
import csv
import os
import subprocess
import sys
import tempfile
import traceback

# Sibling module for the batch command. Fusion does not reliably put the add-in's own
# folder on sys.path, and an ImportError at module scope takes the WHOLE add-in down
# (the single-tile command included), so put the folder on the path explicitly and let
# a failure here degrade to "batch unavailable" rather than "add-in won't load".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import BatchReview
    _BATCH_ERR = None
except Exception:
    BatchReview = None
    _BATCH_ERR = traceback.format_exc()

# ── CONFIG — adjust to your machine ─────────────────────────────────────────────
try:
    import hl_config
except Exception:                     # add-in must still load without it
    hl_config = None

def _cfg(name, default):
    """local_config.py > HELMET_LATTICE_<name> env var > default."""
    return hl_config.get(name, default) if hl_config else default

REPO_DIR   = _cfg("REPO_DIR",   r"C:\Users\Kyrio\helmet-lattice")
PYTHON_EXE = _cfg("PYTHON_EXE", r"C:\Users\Kyrio\AppData\Local\Programs\Python\Python313\python.exe")
DATA_CSV   = _cfg("DATA_CSV",   "")   # "" -> pipeline default (data/raw/dummy_master.csv)
OPT_CANDIDATES = 1000   # Sobol pool for the autonomous optimizer (smaller = faster)
# ────────────────────────────────────────────────────────────────────────────────

CMD_ID   = "helmetLatticeGen"
CMD_NAME = "Helmet Lattice Tile"
CMD_DESC = "Generate a TPU helmet-liner lattice tile as a mesh body."
PANEL_ID = "SolidCreatePanel"
PROPOSALS_CSV = os.path.join(REPO_DIR, "outputs", "proposed_designs.csv")

TOPOLOGIES = ("gyroid", "schwartz", "diamond", "graded",
              "auxetic", "alternating", "blend")
BLEND_BASIS = ("gyroid", "schwartz", "diamond")

app = adsk.core.Application.get()
ui = app.userInterface
_handlers = []   # keep handler refs alive — Fusion GC's them otherwise


def _spinner(inputs, sid, name, val, mn, mx, step):
    """Unitless float spinner (we label units in the name to dodge cm/mm conversion)."""
    return inputs.addFloatSpinnerCommandInput(sid, name, "", mn, mx, step, val)


def _selected(inputs, sid):
    return inputs.itemById(sid).selectedItem.name


def _select_dropdown(dd, name):
    for it in dd.listItems:
        it.isSelected = (it.name == name)


def _apply_visibility(inputs):
    """Show only the controls relevant to the current mode/topology."""
    auto = inputs.itemById("auto").value
    topo = _selected(inputs, "topology")
    # In autonomous mode the optimizer chooses topology/blend/density/cell from data,
    # so hide the manual design controls and keep only the generation knobs.
    for sid in ("topology", "loadProp", "cell", "density"):
        inputs.itemById(sid).isVisible = not auto
    inputs.itemById("gradient").isVisible = (not auto) and topo == "graded"
    inputs.itemById("layers").isVisible = (not auto) and topo == "alternating"
    inputs.itemById("blendGrp").isVisible = (not auto) and topo == "blend"


def _run_optimizer():
    """Run the data-driven optimizer (GP surrogate + Bayesian opt over the TPMS blend
    space) to write a fresh proposed_designs.csv. Returns True on success."""
    script = os.path.join(REPO_DIR, "scripts", "run_pipeline.py")
    cmd = [PYTHON_EXE, script, "--generative", "--no-stl",
           "--candidates", str(OPT_CANDIDATES)]
    if DATA_CSV:
        cmd += ["--data", DATA_CSV]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=0x08000000, cwd=REPO_DIR)
    if proc.returncode != 0:
        ui.messageBox("Optimizer run failed.\n\nstdout:\n" + proc.stdout[-1500:]
                      + "\n\nstderr:\n" + proc.stderr[-1500:])
        return False
    return True


def _load_latest_proposal(inputs, silent=False):
    """Read the last row of proposed_designs.csv and pre-fill the dialog."""
    if not os.path.exists(PROPOSALS_CSV):
        ui.messageBox(f"No proposals file found at:\n{PROPOSALS_CSV}\n\n"
                      "Run scripts/run_pipeline.py first.")
        return
    with open(PROPOSALS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        ui.messageBox("proposed_designs.csv has no rows.")
        return
    r = rows[-1]

    def setf(sid, key):
        try:
            inputs.itemById(sid).value = float(r[key])
        except (KeyError, ValueError, TypeError):
            pass

    setf("cell", "cell_size_mm")
    setf("density", "rel_density_measured")
    setf("gradient", "gradient_param")

    blend = r.get("blend", "")
    if blend and str(blend).strip() and str(blend).strip().lower() != "nan":
        try:
            w = ast.literal_eval(blend)            # e.g. {'gyroid':0.9,'schwartz':0.1}
            _select_dropdown(inputs.itemById("topology"), "blend")
            for b in BLEND_BASIS:
                inputs.itemById(f"w_{b}").value = float(w.get(b, 0.0))
        except Exception:
            pass
    else:
        topo = str(r.get("topology", "")).strip()
        if topo in TOPOLOGIES:
            _select_dropdown(inputs.itemById("topology"), topo)

    _apply_visibility(inputs)
    if not silent:
        ui.messageBox("Loaded latest proposal:\n\n"
                      f"topology/blend: {r.get('blend') or r.get('topology')}\n"
                      f"rho={r.get('rel_density_measured')}, cell={r.get('cell_size_mm')}")


class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            cmd.okButtonText = "Generate"
            inputs = cmd.commandInputs

            topo = inputs.addDropDownCommandInput(
                "topology", "Topology", adsk.core.DropDownStyles.TextListDropDownStyle)
            for i, t in enumerate(TOPOLOGIES):
                topo.listItems.add(t, i == 0)

            inputs.addBoolValueInput(
                "auto", "Autonomous: optimize blend from data", True, "", False)
            inputs.addBoolValueInput("loadProp", "Load latest proposal", True, "", False)

            _spinner(inputs, "cell", "Cell size (mm)", 8.0, 2.0, 30.0, 0.5)
            _spinner(inputs, "density", "Relative density", 0.27, 0.10, 0.55, 0.01)
            _spinner(inputs, "gradient", "Gradient spread (graded)", 0.17, 0.0, 0.4, 0.01)
            _spinner(inputs, "pitch", "Voxel pitch (mm) — smaller=finer/slower", 0.5, 0.2, 1.5, 0.1)

            inputs.addStringValueInput(
                "layers", "Layers (alternating)", "gyroid:0.25:8,schwartz:0.30:6")

            # Generative blend weights (auto-normalized); shown only for 'blend'.
            bg = inputs.addGroupCommandInput("blendGrp", "Blend weights (generative)")
            bg.isExpanded = True
            bc = bg.children
            _spinner(bc, "w_gyroid", "gyroid", 1.0, 0.0, 1.0, 0.05)
            _spinner(bc, "w_schwartz", "schwartz", 0.0, 0.0, 1.0, 0.05)
            _spinner(bc, "w_diamond", "diamond", 0.0, 0.0, 1.0, 0.05)

            grp = inputs.addGroupCommandInput("sizeGrp", "Tile size")
            grp.isExpanded = False
            si = grp.children
            _spinner(si, "sx", "X (mm)", 50.8, 10.0, 200.0, 1.0)
            _spinner(si, "sy", "Y (mm)", 50.8, 10.0, 200.0, 1.0)
            _spinner(si, "sz", "Z (mm)", 38.1, 10.0, 200.0, 1.0)

            _apply_visibility(inputs)

            onChanged = InputChangedHandler()
            cmd.inputChanged.add(onChanged)
            _handlers.append(onChanged)

            onExec = CommandExecuteHandler()
            cmd.execute.add(onExec)
            _handlers.append(onExec)
        except Exception:
            ui.messageBox("CommandCreated failed:\n" + traceback.format_exc())


class InputChangedHandler(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            inputs = args.inputs
            cid = args.input.id
            if cid in ("topology", "auto"):
                _apply_visibility(inputs)
            elif cid == "loadProp" and args.input.value:
                _load_latest_proposal(inputs)
                args.input.value = False        # reset the one-shot button
        except Exception:
            ui.messageBox("InputChanged failed:\n" + traceback.format_exc())


class CommandExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs

            # Autonomous mode: run the data-driven optimizer, then load its choice
            # into the dialog and generate it — no manual blend weights.
            if inputs.itemById("auto").value:
                if not _run_optimizer():
                    return
                _load_latest_proposal(inputs, silent=True)

            topo = _selected(inputs, "topology")
            cell = inputs.itemById("cell").value
            density = inputs.itemById("density").value
            gradient = inputs.itemById("gradient").value
            pitch = inputs.itemById("pitch").value
            sx = inputs.itemById("sx").value
            sy = inputs.itemById("sy").value
            sz = inputs.itemById("sz").value

            script = os.path.join(REPO_DIR, "scripts", "make_stl.py")
            out = os.path.join(tempfile.gettempdir(),
                               f"fusion_lattice_{topo}_rho{density:.2f}_cell{cell:.1f}.stl")

            cmd = [PYTHON_EXE, script,
                   "--topology", topo,
                   "--cell", f"{cell}",
                   "--density", f"{density}",
                   "--gradient", f"{gradient}",
                   "--pitch", f"{pitch}",
                   "--size-x", f"{sx}", "--size-y", f"{sy}", "--size-z", f"{sz}",
                   "--out", out]

            if topo == "blend":
                wg = inputs.itemById("w_gyroid").value
                ws = inputs.itemById("w_schwartz").value
                wd = inputs.itemById("w_diamond").value
                if (wg + ws + wd) <= 0:
                    ui.messageBox("Blend weights are all zero — set at least one.")
                    return
                cmd += ["--blend", f"gyroid:{wg},schwartz:{ws},diamond:{wd}"]
            elif topo == "alternating":
                layers = inputs.itemById("layers").value.strip()
                if layers:
                    cmd += ["--layers", layers]

            # CREATE_NO_WINDOW (0x08000000) keeps a console from flashing up.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  creationflags=0x08000000)
            if proc.returncode != 0 or not os.path.exists(out):
                ui.messageBox("Lattice generation failed.\n\n"
                              f"cmd: {' '.join(cmd)}\n\n"
                              f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")
                return

            design = adsk.fusion.Design.cast(app.activeProduct)
            if not design:
                ui.messageBox("No active Fusion design. Open a design and try again.")
                return
            root = design.rootComponent

            # STL is written in mm. Mesh import in a parametric design must happen
            # inside a base-feature edit; direct-modeling designs can add directly.
            units = adsk.fusion.MeshUnits.MillimeterMeshUnit
            if design.designType == adsk.fusion.DesignTypes.ParametricDesignType:
                base = root.features.baseFeatures.add()
                base.startEdit()
                try:
                    root.meshBodies.add(out, units, base)
                finally:
                    base.finishEdit()
            else:
                root.meshBodies.add(out, units)

            ui.messageBox("Imported lattice tile.\n\n" + proc.stdout.strip())
        except Exception:
            ui.messageBox("Execute failed:\n" + traceback.format_exc())


def run(context):
    try:
        cmdDef = ui.commandDefinitions.itemById(CMD_ID)
        if not cmdDef:
            cmdDef = ui.commandDefinitions.addButtonDefinition(CMD_ID, CMD_NAME, CMD_DESC)

        onCreated = CommandCreatedHandler()
        cmdDef.commandCreated.add(onCreated)
        _handlers.append(onCreated)

        panel = ui.allToolbarPanels.itemById(PANEL_ID)
        if panel and not panel.controls.itemById(CMD_ID):
            ctrl = panel.controls.addCommand(cmdDef)
            # addCommand() alone buries the entry at the bottom of the CREATE
            # dropdown; promoting puts it on the panel itself where it's findable.
            ctrl.isPromoted = True
            ctrl.isPromotedByDefault = True

        # Second command: batch generation into its own Data Panel folder. Isolated so
        # the single-tile command still loads if the batch module has a problem.
        if BatchReview is not None:
            try:
                BatchReview.add_to_panel()
            except Exception:
                ui.messageBox("Batch command failed to register (the single-tile "
                              "command still works):\n" + traceback.format_exc())
        else:
            ui.messageBox("Batch command unavailable — BatchReview.py failed to "
                          "import (the single-tile command still works):\n\n"
                          + (_BATCH_ERR or ""))
    except Exception:
        ui.messageBox("Add-in run() failed:\n" + traceback.format_exc())


def stop(context):
    try:
        panel = ui.allToolbarPanels.itemById(PANEL_ID)
        if panel:
            ctrl = panel.controls.itemById(CMD_ID)
            if ctrl:
                ctrl.deleteMe()
        cmdDef = ui.commandDefinitions.itemById(CMD_ID)
        if cmdDef:
            cmdDef.deleteMe()
        if BatchReview is not None:
            BatchReview.remove_from_panel()
    except Exception:
        ui.messageBox("Add-in stop() failed:\n" + traceback.format_exc())
