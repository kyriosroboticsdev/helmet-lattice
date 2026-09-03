"""
BatchReview — Fusion 360 batch generation for human review
==========================================================
Generates N optimizer-proposed lattice designs in one go and saves each as its own
Fusion document inside a new Data Panel folder, ready to open and review.

How it works
------------
Same split as HelmetLattice.py: Fusion's bundled Python has no numpy/scikit-image/
trimesh/torch, so the ML + meshing runs in the *system* Python via

    <PYTHON_EXE>  <REPO_DIR>/scripts/run_pipeline.py --generative --q N

which writes outputs/proposed_designs.csv (one row per accepted design, each with an
`stl_path`). This module then imports each STL into a fresh document and saves it into
a new folder in the active Fusion Teams project.

The mesh you review IS the mesh you print -- there is no separate preview tier. One
mesh per document keeps that affordable: the alternative (all N in one document) is
what makes full-resolution meshes unmanageable, not the resolution itself.

A "folder" here is a Fusion Teams / Data Panel folder -- cloud storage in the active
project, not a folder on the local disk. The STLs stay on disk in stl_output/ and are
what you hand to the slicer.
"""

import adsk.core
import adsk.fusion
import ast
import csv
import datetime
import os
import subprocess
import traceback

# ── CONFIG — adjust to your machine (mirrors HelmetLattice.py) ─────────────────
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
OPT_CANDIDATES = 1000    # Sobol pool for the optimizer (smaller = faster)
# ───────────────────────────────────────────────────────────────────────────────

CMD_ID   = "helmetLatticeBatch"
CMD_NAME = "Helmet Lattice Batch"
CMD_DESC = ("Generate N optimizer-proposed lattice designs and save each as its own "
            "document in a new folder for review.")
PANEL_ID = "SolidCreatePanel"
PROPOSALS_CSV = os.path.join(REPO_DIR, "outputs", "proposed_designs.csv")

app = adsk.core.Application.get()
ui = app.userInterface
_handlers = []   # keep handler refs alive — Fusion GC's them otherwise


def _fmt_blend(blend_str):
    """'{'diamond':0.45,...}' -> 'dia45-re_30-vor25' for a compact document name."""
    try:
        w = ast.literal_eval(blend_str)
    except Exception:
        return "design"
    return "-".join(f"{k[:3]}{int(round(v * 100))}"
                    for k, v in sorted(w.items(), key=lambda kv: -kv[1]))


def _run_batch(n, progress):
    """Run the optimizer + mesher for n designs. Returns the accepted proposal rows.

    --q n asks the pipeline for n designs that clear both the descriptor-space
    diversity filter and the printability gate; it ranks a 3x shortlist internally so
    gate rejections get replaced rather than shrinking the batch.
    """
    script = os.path.join(REPO_DIR, "scripts", "run_pipeline.py")
    cmd = [PYTHON_EXE, script, "--generative", "--q", str(n),
           "--candidates", str(OPT_CANDIDATES)]
    if DATA_CSV:
        cmd += ["--data", DATA_CSV]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=0x08000000, cwd=REPO_DIR)
    if proc.returncode != 0:
        ui.messageBox("Batch generation failed.\n\nstdout:\n" + proc.stdout[-1500:]
                      + "\n\nstderr:\n" + proc.stderr[-1500:])
        return None

    if not os.path.exists(PROPOSALS_CSV):
        ui.messageBox(f"Pipeline finished but wrote no proposals file:\n{PROPOSALS_CSV}")
        return None
    with open(PROPOSALS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("stl_path") and os.path.exists(r["stl_path"])]
    if not rows:
        ui.messageBox("Pipeline produced no designs with a usable STL.\n\n"
                      + proc.stdout[-1500:])
        return None
    return rows


def _target_folder():
    """Create a timestamped folder in the active project's root. Returns (folder, name).

    Fusion documents live in a Fusion Teams project, so this is cloud storage -- the
    folder appears in the Data Panel, not on the local disk.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"LatticeBatch_{stamp}"
    project = app.data.activeProject
    if not project:
        ui.messageBox("No active Fusion project. Open the Data Panel and select a "
                      "project, then run this again.")
        return None, name
    return project.rootFolder.dataFolders.add(name), name


def _import_and_save(row, folder, index, total, progress):
    """Open a new document, import the design's mesh, save it into `folder`."""
    stl = row["stl_path"]
    blend = row.get("blend") or ""
    tag = _fmt_blend(blend) if blend.strip() and blend.strip().lower() != "nan" \
        else str(row.get("topology", "design"))
    doc_name = f"D{index:02d}_{tag}"

    progress.message = f"Importing {index}/{total}: {doc_name}"

    # A new document starts in direct-modeling ("DirectDesignType") or parametric
    # depending on the user's preference; mesh import in a parametric design has to
    # happen inside a base-feature edit, so handle both rather than assume.
    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    units = adsk.fusion.MeshUnits.MillimeterMeshUnit
    if design.designType == adsk.fusion.DesignTypes.ParametricDesignType:
        base = root.features.baseFeatures.add()
        base.startEdit()
        try:
            root.meshBodies.add(stl, units, base)
        finally:
            base.finishEdit()
    else:
        root.meshBodies.add(stl, units)

    # Name the component after the design so the browser is readable, and put the
    # numbers that matter for review into the document description.
    try:
        root.name = doc_name
    except Exception:
        pass

    desc = (f"blend: {blend}\n"
            f"rho designed={row.get('rel_density_measured')}, "
            f"as-built={row.get('rel_density_as_built')}\n"
            f"cell={row.get('cell_size_mm')} mm, "
            f"families={row.get('n_families')}, "
            f"max share={row.get('max_family_weight')}\n"
            f"pred HIC={row.get('pred_HIC')}, pred BrIC={row.get('pred_BrIC')}\n"
            f"unsupported area={row.get('unsupported_frac')}\n"
            f"STL: {stl}")
    doc.saveAs(doc_name, folder, desc, "")
    return doc_name


class ExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            inputs = args.command.commandInputs
            n = int(inputs.itemById("count").value)

            progress = ui.createProgressDialog()
            progress.isBackgroundTranslucent = False
            progress.show("Helmet Lattice Batch",
                          "Running optimizer + meshing %v of %m ...", 0, n + 1, 0)
            progress.progressValue = 0
            progress.message = ("Running optimizer and meshing %d designs — this is "
                                "the slow step, Fusion will be busy." % n)

            rows = _run_batch(n, progress)
            if rows is None:
                progress.hide()
                return
            progress.progressValue = 1

            folder, folder_name = _target_folder()
            if folder is None:
                progress.hide()
                return

            made = []
            for i, row in enumerate(rows, start=1):
                if progress.wasCancelled:
                    break
                made.append(_import_and_save(row, folder, i, len(rows), progress))
                progress.progressValue = 1 + i
            progress.hide()

            ui.messageBox(
                f"Saved {len(made)} design(s) to Data Panel folder:\n"
                f"  {folder_name}\n\n" + "\n".join("  " + m for m in made) +
                "\n\nThe STLs for printing are in stl_output/ — the mesh in each "
                "document is the same file.")
        except Exception:
            ui.messageBox("Batch execute failed:\n" + traceback.format_exc())


class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inputs = cmd.commandInputs
            inputs.addTextBoxCommandInput(
                "info", "", "Runs the optimizer, meshes each accepted design, and "
                "saves each as its own document in a new Data Panel folder. "
                "Expect roughly a minute per design.", 3, True)
            inputs.addIntegerSpinnerCommandInput("count", "Designs", 1, 25, 1, 10)

            onExec = ExecuteHandler()
            cmd.execute.add(onExec)
            _handlers.append(onExec)
        except Exception:
            ui.messageBox("Batch command setup failed:\n" + traceback.format_exc())


def add_to_panel():
    """Register the batch command. Called from HelmetLattice.run()."""
    cmdDef = ui.commandDefinitions.itemById(CMD_ID)
    if not cmdDef:
        cmdDef = ui.commandDefinitions.addButtonDefinition(CMD_ID, CMD_NAME, CMD_DESC)
    onCreated = CommandCreatedHandler()
    cmdDef.commandCreated.add(onCreated)
    _handlers.append(onCreated)
    panel = ui.allToolbarPanels.itemById(PANEL_ID)
    if panel and not panel.controls.itemById(CMD_ID):
        ctrl = panel.controls.addCommand(cmdDef)
        # Promote so the batch command sits on the CREATE panel itself rather than
        # at the bottom of its dropdown, where it is easy to miss.
        ctrl.isPromoted = True
        ctrl.isPromotedByDefault = True


def remove_from_panel():
    """Unregister. Called from HelmetLattice.stop()."""
    panel = ui.allToolbarPanels.itemById(PANEL_ID)
    if panel:
        ctrl = panel.controls.itemById(CMD_ID)
        if ctrl:
            ctrl.deleteMe()
    cmdDef = ui.commandDefinitions.itemById(CMD_ID)
    if cmdDef:
        cmdDef.deleteMe()
