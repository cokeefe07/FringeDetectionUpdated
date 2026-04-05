# SpellBook — Fringe Detection & Overlay App

SpellBook is a Tkinter-based desktop tool for:
1. Preprocessing (even illumination / contrast enhancement)
2. Automatic fringe (line) detection & skeletonization
3. Manual mask touch‑up (add/remove/link endpoints)
4. Overlay visualization for quality review

> **Note:** A detailed user manual is available in `help.html`.

---
## 1. Repository Layout

| Path | Purpose |
|------|---------|
| `SpellBook.py` | Entry point launching the GUI (Overlay, Detection, Editor tabs). |
| `fringe_detection/` | Processing helpers: shading pipeline, binarization & oriented opening utilities. |
| `tabs/overlay_tab.py` | Overlay/registration & cropping UI. |
| `tabs/fringe_editor.py` | Interactive binary mask editor (paint + link endpoints). |
| `mixins/viewport_rendering.py` | Shared viewport zoom/pan helpers (legacy mixin). |
| `requirements.txt` | Python dependencies. |

---
## 2. Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

macOS / Linux (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

If wheel installation fails (e.g. on Apple Silicon), use a Conda/Miniforge environment.

---
## 3. Tutorial Walkthrough

### Step 0: Prepare Your Images
Place raw input images in a convenient folder. Supported formats: PNG / TIFF / JPG. High bit‑depth TIFFs are internally normalized to 8‑bit for display.

### Step 1: Launch & Load (Detection Tab)
1. Click `Browse & Load` to pick an image.
2. Adjust preprocessing sliders:
   - Blur σ: mild Gaussian smoothing for noise reduction.
   - CLAHE clip & tile: local contrast expansion; high clip increases contrast, tile defines grid size.
3. Original overlay (optional): blend some of the original intensity back into processed views for context.

The two center viewers show:
| Viewer | Content |
|--------|---------|
| Illumination | Enhanced grayscale (after shading + contrast). |
| Fringe Overlay | Detected fringe skeleton overlayed (green/colored) on processed background. |

Zoom: mouse wheel (each viewer independent). Pan: right‑click drag. Status bar shows current zoom.

### Step 2: Tune Fringe Detection
Right panel sliders:
| Slider | Effect |
|--------|--------|
| Kernel length | Line structuring element length for oriented opening (line extraction). |
| Kernel thickness | Line thickness assumption. |
| Angle ± / Angle step | Angular sweep around horizontal for oriented opening (coverage vs. speed). |
| Dilate px | Thickens detected ridges before skeletonization. |
| Min area | Removes small specks before skeletonizing. |
| Prune spurs | Removes small branches from the skeletonized fringes. |
| Fill loops | Closes small holes within fringes. |
| Hump Width | Flattens fringes by removing "humps" (0 = off). |
| Background opacity | Adjusts the visibility of the original image behind the detected fringes. |

After adjustments the overlay viewer updates automatically (debounced ~180 ms).

### Step 3: Overlay & Registration (Overlay Tab)
Switch to `Overlay` to align a "Shot" image with a "Reference" image:
1. Load Reference and Shot images.
2. Use **Move Both** (Right-click drag) or **Move Shot** (Left-click drag) to align.
3. Use arrow keys or on-screen buttons for 1px nudge precision.
4. Adjust **Shot Opacity** (default 0.5) to check alignment.
5. Use **Crop Mode** to define a region of interest and crop both images simultaneously.
6. Save the aligned images for further processing.

### Step 4: Save Automatic Result
Click `Save Fringes as Binary` in the  to export the current binary mask (0 = fringe, 255 = background).

### Step 5: Fine Editing (Editor Tab)
Switch to `Editor`:
1. `Open Binary` – load a saved mask OR create one from an image.
2. (Optional) `Open Background` – load a grayscale backdrop.
3. **Brush Modes**:
   - `Add/Remove Black`: Draw or erase binary fringe lines.
   - `Add/Remove Mask (Gray)`: Paint a gray mask to exclude regions.
4. **Mouse**:
   - Left drag: paint / erase.
   - Ctrl + wheel: change brush radius.
   - Wheel: zoom.
   - Right drag: pan.
5. **Tools**:
   - `Link endpoints`: Connect nearby skeleton endpoints within tolerance & angular constraint.
   - `Magic2 Tester`: Draws an invisible vertical line to highlight touching fringes (useful for splitting).
   - `Color comps`: Pseudo‑colors connected components.
   - `Overlay Binary` / `Merge Overlay`: Combine another binary mask with the current one.
6. **Visualization**:
   - `Background brightness`: Adjust background intensity.
   - `Fringe Opacity`: Adjust fringe overlay visibility.
7. `Undo` reverts last stroke.

No mask is auto‑loaded into the Editor; you decide when to load or import one.

### Step 6: Iterate & Export
Refine, then `Save As…` in the Editor for a cleaned fringe mask. Use saved masks for downstream analysis or comparison.

---
## 4. Design Notes
| Aspect | Choice |
|--------|--------|
| Independent zoom | Each viewer maintains its own zoom state for local inspection. |
| Skeletonization | Uses `skimage.morphology.skeletonize` on prefiltered binary. |
| Oriented opening | Sweeps discrete angles (± range, given step) with line structuring elements for ridge isolation. |
| Endpoint linking | Brute-force within radius + angle gate + component separation. |
| Performance | Debounced slider changes; uses integer structuring elements. |

---
## 5. Building a Standalone Executable (Windows)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --name "SpellBook" SpellBook.py
```
Binary appears in `dist/SpellBook.exe`.

GitHub Actions can be configured to run this on tag push (e.g. `v1.2.0`).

---
## 6. Troubleshooting
| Issue | Fix |
|-------|-----|
| Wheel zoom not working | Ensure window focus; on Linux use `<Button-4>/<Button-5>` events. |
| Missing DLL (OpenCV) | Reinstall with `pip install --force-reinstall opencv-python`. |
| Slow large images | Reduce Blur σ & CLAHE tile; disable Color comps during editing. |
| No fringes detected | Increase kernel length, lower Min area, adjust Angle ±. |

---
## 7. Contributing
1. Fork & branch: `git checkout -b feature/xyz`.
2. Keep UI changes minimal per commit.
3. Run lint/tests (add if missing) before PR.

---
## 8. License & Contact
Add `LICENSE` (MIT recommended) and maintainer contact/email here.

---
## 9. At-a-Glance Commands
```powershell
# Setup
python -m venv .venv; ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Run
python SpellBook.py

# Build executable
pyinstaller --onefile --name SpellBook SpellBook.py
```

---
## 10. FAQ
**Q: Why two viewers?** Independent inspection of raw enhancement vs. overlay.
**Q: Why isn’t the Editor auto-filled?** Manual control prevents accidental edits; load explicitly.
**Q: Units of link tolerance?** Pixels in original image coordinates.

---
Happy detecting and editing! Feel free to open issues for feature requests.
