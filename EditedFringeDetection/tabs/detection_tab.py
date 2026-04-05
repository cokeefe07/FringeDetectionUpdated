import os
import threading
import types
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

from fringe_detection import (
    pipeline_shading_sauvola,
    read_gray,
    binarize,
    oriented_opening,
    overlay_mask_on_gray,
    make_slider_row,
    to_photoimage_from_bgr_with_scale,
    remove_humps,
    remove_branches,
    fill_holes,
    remove_steep_segments,
)
from fringe_detection.zoom_handler import ZoomPanHandler


class DetectionTabFrame(ttk.Frame):
    """Detection tab: preprocess image and display illumination + fringe overlay."""

    def __init__(self, master, status_callback=None):
        super().__init__(master)
        self._status_cb = status_callback or (lambda txt: None)

    # Data/state
        self.src_img = None
        self.enh_img = None
        self._after_id = None
    # Render scheduling
        self._render_running = False
        self._pending_params = None
        
    # Caching for shading pipeline
        self._last_shading_params = None
        self._cached_enh_img = None

    # Viewer state
        self._illum_img_id = None
        self._fringe_img_id = None
        self._photo_illum = None
        self._photo_fringe = None
        self._last_illum_bgr = None
        self._last_overlay_bgr = None
        self._illum_zoom = 1.0
        self._fringe_zoom = 1.0
        self._zoom_level = 1.0
        self._illum_centered = False
        self._fringe_centered = False

        self._build_ui()
        self._attach_handlers()

    # UI helpers
    def _make_slider_row(self, parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
        if command is None:
            command = self.on_param_change
        return make_slider_row(parent, label_text, var, frm, to, resolution=resolution, is_int=is_int, fmt=fmt, command=command)

    def _build_ui(self):
        # Left control panel container
        ctrl_container = ttk.Frame(self)
        ctrl_container.pack(side='left', fill='y', padx=8, pady=8)
        ctrl_container.config(width=260)
        ctrl_container.pack_propagate(False)

        # Canvas and Scrollbar for scrolling
        try:
            bg_color = self.cget('background')
            if not bg_color: bg_color = '#f0f0f0'
        except Exception:
            bg_color = '#f0f0f0'
            
        canvas = tk.Canvas(ctrl_container, highlightthickness=0, bg=bg_color)
        scrollbar = ttk.Scrollbar(ctrl_container, orient="vertical", command=canvas.yview)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas.configure(yscrollcommand=scrollbar.set)

        # The actual control frame
        ctrl = ttk.Frame(canvas)
        
        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=ctrl, anchor="nw")

        def check_scrollbar():
            bbox = canvas.bbox("all")
            if not bbox: return
            _, _, _, height = bbox
            if height > canvas.winfo_height():
                scrollbar.pack(side="right", fill="y")
            else:
                scrollbar.pack_forget()

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            check_scrollbar()
        
        ctrl.bind("<Configure>", configure_scroll_region)

        def configure_window_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
            check_scrollbar()

        canvas.bind("<Configure>", configure_window_width)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"

        def _recursive_bind(widget):
            widget.bind('<MouseWheel>', _on_mousewheel)
            for child in widget.winfo_children():
                _recursive_bind(child)

        # Bind canvas and schedule recursive bind for ctrl (to catch all children added later)
        canvas.bind('<MouseWheel>', _on_mousewheel)
        self.after_idle(lambda: _recursive_bind(ctrl))

        left_title = ttk.Frame(ctrl)
        left_title.pack(anchor='w', fill='x')
        ttk.Label(left_title, text='Fringe Detection and Clean Up', font=('Segoe UI', 10, 'bold')).pack(side='left')

        def make_help_icon(parent, tooltip_text, side='right'):
            try:
                bg = self.cget('background')
            except Exception:
                bg = '#f0f0f0'
            c = tk.Canvas(parent, width=18, height=18, highlightthickness=0, bg=bg)
            c.create_oval(2, 2, 16, 16, outline='#666', width=1)
            c.create_text(9, 9, text='?', font=('Segoe UI', 9))
            c.pack(side='left', padx=(6, 0))
            self._attach_tooltip(c, tooltip_text, side=side)
            return c

        make_help_icon(left_title, (
            'Detection Tab Purpose:\n'
            'This tab illuminates the image for better fringe\n'
            'detection then detects and traces the fringes \n'
            '\n'
            'Controls:\n'
            '- Right-click drag to move image\n'
            '- Mouse wheel to zoom\n'
            '\n'
            'Fringe Detection Features:\n'
            '- Sharpness: Changes sharpness of the binary image\n'
            '- Line length: Adjusts the length of detected fringes\n'
            '- Line thickness: Adjusts the thickness of detected fringes\n'
            '- Max angle°: Adjusts the maximum angle for fringe detection\n'
            '- Dialte(px): Adjusts the dilation for fringe detection\n'
            '- Min area: Adjusts the minimum area for fringe detection\n'
            '- Prune spurs: Removes small branches from fringes\n'
            '- Fill loops: Removes small loops/circles from fringes\n'
            '- Hump Width: Flattens Fringes by removing humps (0 = off)\n'
            '- Background opacity: Adjusts the background opacity for better fringe viewing\n'
            ))

        # Row with Browse and Save
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(anchor='w', pady=4)
        ttk.Button(btn_row, text='Load Image', command=self.load_image_dialog).pack(side='left')
        ttk.Button(btn_row, text='Save Fringes as Binary', command=self.save_result).pack(side='left', padx=(6, 0))

        # Edit sliders (button moved to end)

        # Initialize slider metadata and default vars before creating the Background opacity slider
        self.sigma_var = tk.DoubleVar(value=7.0)
        self.clip_var = tk.DoubleVar(value=100.0)
        self.tile_var = tk.IntVar(value=8)
        self.win_var = tk.IntVar(value=31)
        self.k_var = tk.DoubleVar(value=0.20)
        self.post_var = tk.IntVar(value=1)
        self._slider_meta = {}

        # Background opacity slider placed near load/save
        self.bg_opacity = tk.DoubleVar(value=1.0)
        s_bg = self._make_slider_row(ctrl, 'Background opacity', self.bg_opacity, 0.0, 1.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['Background opacity'] = {'scale': s_bg, 'var': self.bg_opacity, 'is_int': False, 'frm': 0.0, 'to': 1.0}

        self.status = ttk.Label(ctrl, text='Ready', wraplength=110)
        self.status.pack(anchor='w', pady=(0, 6))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        
        ttk.Label(ctrl, text='Fringe Detection', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(4, 2))
        s1 = self._make_slider_row(ctrl, 'Background Image Sharpness', self.sigma_var, 0.1, 10.0, is_int=False, fmt="{:.1f}")
        self._slider_meta['Background Image Sharpness'] = {'scale': s1, 'var': self.sigma_var, 'is_int': False, 'frm': 0.1, 'to': 10.0}

        # --- NEW: Background Clip Slider ---
        self.bg_clip_var = tk.IntVar(value=255)
        s_bg_clip = self._make_slider_row(ctrl, 'Background Clip (White Point)', self.bg_clip_var, 0, 255, is_int=True)
        self._slider_meta['Background Clip (White Point)'] = {'scale': s_bg_clip, 'var': self.bg_clip_var, 'is_int': True, 'frm': 0, 'to': 255}
        # -----------------------------------
        # (Sauvola win/k/post open removed; values still used internally)
        
        # Fringe detection parameters
        self.k_len = tk.IntVar(value=20)
        self.k_thk = tk.IntVar(value=2)
        self.k_ang = tk.DoubleVar(value=5.0)
        self.k_dilate = tk.IntVar(value=2)
        self.k_area = tk.IntVar(value=0)
        # Auto-compute Min area from length/thickness/dilate (always enabled)
        self.k_area_auto = tk.BooleanVar(value=True)
        # Hump removal
        self.hump_width_var = tk.IntVar(value=0)
        # Branch pruning
        self.prune_var = tk.IntVar(value=0)
        # Fill loops
        self.fill_loops_var = tk.IntVar(value=0)
        # Dust removal
        self.dust_var = tk.IntVar(value=0)
        # Min fragment feature removed; fragment cleanup handled by other settings
        # Background opacity for Fringe view: 0 => white background, 1 => full grayscale background

        s7 = self._make_slider_row(ctrl, 'Minimum Fringe Length', self.k_len, 5, 200, is_int=True)
        self._slider_meta['Minimum Fringe Length'] = {'scale': s7, 'var': self.k_len, 'is_int': True, 'frm': 5, 'to': 200}
        s8 = self._make_slider_row(ctrl, 'Minimum Fringe Thickness', self.k_thk, 1, 8, is_int=True)
        self._slider_meta['Minimum Fringe Thickness'] = {'scale': s8, 'var': self.k_thk, 'is_int': True, 'frm': 1, 'to': 8}
        s9 = self._make_slider_row(ctrl, 'Max angle°', self.k_ang, 0.0, 90.0, is_int=False, fmt="{:.1f}")
        self._slider_meta['Max angle°'] = {'scale': s9, 'var': self.k_ang, 'is_int': False, 'frm': 0.0, 'to': 90.0}
        s11 = self._make_slider_row(ctrl, 'Dilate (px)', self.k_dilate, 0, 25, is_int=True)
        self._slider_meta['Dilate (px)'] = {'scale': s11, 'var': self.k_dilate, 'is_int': True, 'frm': 0, 'to': 25}
        # Min area is now always computed automatically from length/thickness/dilate
        # Keep internal var `self.k_area` to store the computed value but remove slider UI.

        # Post Detection Clean-Up
        ttk.Label(ctrl, text='Post Detection Clean-Up', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(6, 2))

        s_prune = self._make_slider_row(ctrl, 'Prune spurs', self.prune_var, 0, 30, is_int=True)
        self._slider_meta['Prune spurs'] = {'scale': s_prune, 'var': self.prune_var, 'is_int': True, 'frm': 0, 'to': 30}

        s_loops = self._make_slider_row(ctrl, 'Fill loops (px)', self.fill_loops_var, 0, 100, is_int=True)
        self._slider_meta['Fill loops (px)'] = {'scale': s_loops, 'var': self.fill_loops_var, 'is_int': True, 'frm': 0, 'to': 100}

        s_dust = self._make_slider_row(ctrl, 'Remove Dust (px)', self.dust_var, 0, 100, is_int=True)
        self._slider_meta['Remove Dust (px)'] = {'scale': s_dust, 'var': self.dust_var, 'is_int': True, 'frm': 0, 'to': 100}

        # (Min fragment slider removed)

        # Hump removal
        s_hump = self._make_slider_row(ctrl, 'Hump Width', self.hump_width_var, 0, 30, is_int=True)
        self._slider_meta['Hump Width'] = {'scale': s_hump, 'var': self.hump_width_var, 'is_int': True, 'frm': 0, 'to': 30}

        # Background opacity slider for fringe overlay background (moved above)

        # Place 'Edit Sliders' after all controls
        ttk.Button(ctrl, text='Edit Sliders', command=self.open_slider_ranges).pack(anchor='w', pady=(6, 2))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        
        # Extra padding for scrolling
        ttk.Frame(ctrl, height=100).pack(fill='x')

        # Middle viewers
        img_frame = ttk.Frame(self)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        img_frame.pack_propagate(False)

        viewers_container = ttk.Frame(img_frame)
        viewers_container.pack(fill='both', expand=True)

        def make_viewer(parent, title):
            """Viewer canvas (no scrollbars; zoom/pan via handlers)."""
            container = ttk.Frame(parent)
            container.pack(fill='both', expand=True)
            if title:
                ttk.Label(container, text=title, font=('Segoe UI', 9)).pack(anchor='w', padx=2)
            canvas = tk.Canvas(container, bg='black', highlightthickness=0)
            canvas.pack(fill='both', expand=True)
            return canvas

        self.fringe_canvas = make_viewer(viewers_container, '')

    def _attach_handlers(self):
        self._fringe_handler = ZoomPanHandler(
            widget=self.fringe_canvas,
            get_zoom=lambda: self._fringe_zoom,
            set_zoom=lambda z: (setattr(self, '_fringe_zoom', z), setattr(self, '_zoom_level', z)),
            rescale_callback=lambda: self._update_illum_and_fringe(self._last_illum_bgr, self._last_overlay_bgr),
            min_zoom=0.1, max_zoom=64.0, zoom_step=1.1,
        )
        # Linux/X11 wheel shim
        self.fringe_canvas.bind('<Button-4>', lambda e: self._linux_wheel(self._fringe_handler, +1, e))
        self.fringe_canvas.bind('<Button-5>', lambda e: self._linux_wheel(self._fringe_handler, -1, e))

    # Status / tooltip helpers
    def set_status(self, txt):
        # Show status text without zoom suffix
        self.status.config(text=txt)
        try:
            self._status_cb(txt)
        except Exception:
            pass

    def _linux_wheel(self, handler: ZoomPanHandler, direction: int, event):
        fake = types.SimpleNamespace(delta=120 * (1 if direction > 0 else -1), x=event.x, y=event.y)
        try:
            handler._on_mousewheel(fake)
        except Exception:
            pass

    def _attach_tooltip(self, widget, text, side='right'):
        tip = {'win': None}

        def show_tip(_e=None):
            if tip['win'] is not None:
                return
            win = tk.Toplevel(widget)
            tip['win'] = win
            try:
                win.wm_overrideredirect(True)
            except Exception:
                pass
            frame = ttk.Frame(win, borderwidth=1, relief='solid')
            frame.pack()
            lbl = ttk.Label(frame, text=text, justify='left', padding=6)
            lbl.pack()
            try:
                win.update_idletasks()
                wrx = widget.winfo_rootx(); wry = widget.winfo_rooty()
                ww = widget.winfo_width(); wh = widget.winfo_height()
                win_w = win.winfo_width() or win.winfo_reqwidth()
                if str(side).lower() == 'left':
                    x = int(wrx - 8 - win_w)
                else:
                    x = int(wrx + ww + 8)
                y = int(wry + (wh * 0.5))
                if x < 0: x = 0
                if y < 0: y = 0
                win.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass

        def hide_tip(_e=None):
            w = tip.get('win')
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass
                tip['win'] = None

        try:
            widget.bind('<Enter>', show_tip)
            widget.bind('<Leave>', hide_tip)
        except Exception:
            pass

    # Image IO
    def load_image_dialog(self):
        try:
            self.lift()
        except Exception:
            pass
        base_dir = os.path.abspath(os.path.dirname(__file__))
        edited_dir = os.path.join(os.path.dirname(base_dir), 'EditedImages')
        initial_dir = edited_dir if os.path.isdir(edited_dir) else os.path.dirname(base_dir)
        p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=initial_dir,
                                       filetypes=[('Images', ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'))])
        if not p:
            return
        try:
            self.src_img = read_gray(p)
            self.set_status(f'Loaded: {os.path.basename(p)}')
            # Invalidate cache so new image is processed
            self._cached_enh_img = None
            self._last_shading_params = None
            self.start_render_now()
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def save_result(self):
        if hasattr(self, '_binary_mask') and self._binary_mask is not None:
            to_save = self._binary_mask
        else:
            messagebox.showinfo('No result', 'No result to save')
            return
        base_dir = os.path.abspath(os.path.dirname(__file__))
        edited_dir = os.path.join(os.path.dirname(base_dir), 'EditedImages')
        try:
            os.makedirs(edited_dir, exist_ok=True)
        except Exception:
            pass
        p = filedialog.asksaveasfilename(initialdir=edited_dir, defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        try:
            cv2.imwrite(p, to_save)
            self.set_status(f'Saved: {os.path.basename(p)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    # Slider ranges dialog
    def open_slider_ranges(self):
        dlg = tk.Toplevel(self)
        dlg.title('Edit slider ranges')
        dlg.transient(self)
        dlg.grab_set()
        rows = []
        for i, (name, meta) in enumerate(self._slider_meta.items()):
            ttk.Label(dlg, text=name).grid(row=i, column=0, sticky='w', padx=6, pady=4)
            frm_val = tk.StringVar(value=str(meta.get('frm', '')))
            to_val = tk.StringVar(value=str(meta.get('to', '')))
            e1 = ttk.Entry(dlg, textvariable=frm_val, width=8)
            e1.grid(row=i, column=1, padx=4, pady=4)
            e2 = ttk.Entry(dlg, textvariable=to_val, width=8)
            e2.grid(row=i, column=2, padx=4, pady=4)
            rows.append((name, meta, frm_val, to_val))

        def on_apply():
            for name, meta, frm_var, to_var in rows:
                try:
                    if meta['is_int']:
                        new_frm = int(float(frm_var.get()))
                        new_to = int(float(to_var.get()))
                    else:
                        new_frm = float(frm_var.get())
                        new_to = float(to_var.get())
                except Exception:
                    messagebox.showerror('Invalid value', f'Invalid numeric value for {name}')
                    return
                if new_to <= new_frm:
                    messagebox.showerror('Invalid range', f'Max must be > Min for {name}')
                    return
                meta['frm'] = new_frm
                meta['to'] = new_to
                scale = meta.get('scale')
                if scale is not None:
                    try:
                        scale.config(from_=new_frm, to=new_to)
                        var = meta.get('var')
                        if var is not None:
                            v = var.get()
                            if v < new_frm:
                                var.set(new_frm)
                            elif v > new_to:
                                var.set(new_to)
                    except Exception:
                        pass
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_fr = ttk.Frame(dlg)
        btn_fr.grid(row=len(rows), column=0, columnspan=3, pady=8)
        ttk.Button(btn_fr, text='Apply', command=on_apply).pack(side='left', padx=6)
        ttk.Button(btn_fr, text='Cancel', command=on_cancel).pack(side='left', padx=6)

    # Render scheduling
    def on_param_change(self, _=None):
        # Immediate reaction to slider changes (schedule full recompute if needed)
        self.start_render_now()

    def start_render_now(self):
        if self.src_img is None:
            self.set_status('No image loaded')
            return
        # Compute Min area automatically when auto enabled, else use slider value
        area_val = int(self.k_area.get()) if hasattr(self, 'k_area') else 0
        try:
            if hasattr(self, 'k_area_auto') and self.k_area_auto.get():
                eff_w = max(1, int(self.k_thk.get()) + 2 * int(self.k_dilate.get()))
                area_auto = int(round(eff_w * int(self.k_len.get()) * 1.15))
                area_val = area_auto
                try:
                    # reflect computed value in UI slider
                    self.k_area.set(area_auto)
                except Exception:
                    pass
        except Exception:
            area_val = int(self.k_area.get()) if hasattr(self, 'k_area') else 0

        # Include ALL relevant parameters (fringe + shading) so changes during an active render queue a new one.
        params = (
            float(self.sigma_var.get()),
            float(self.clip_var.get()),
            int(self.tile_var.get()),
            int(self.win_var.get()),
            float(self.k_var.get()),
            int(self.post_var.get()),
            int(self.k_len.get()) if hasattr(self, 'k_len') else 0,
            int(self.k_thk.get()) if hasattr(self, 'k_thk') else 0,
            float(self.k_ang.get()) if hasattr(self, 'k_ang') else 0.0,
            int(self.k_dilate.get()) if hasattr(self, 'k_dilate') else 0,
            int(area_val),
            float(self.bg_opacity.get()) if hasattr(self, 'bg_opacity') else 1.0,
            (int(self.hump_width_var.get()) > 0) if hasattr(self, 'hump_width_var') else False,
            int(self.hump_width_var.get()) if hasattr(self, 'hump_width_var') else 2,
            int(self.prune_var.get()) if hasattr(self, 'prune_var') else 0,
            int(self.fill_loops_var.get()) if hasattr(self, 'fill_loops_var') else 0,
            int(self.min_fragment_var.get()) if hasattr(self, 'min_fragment_var') else 0,
            int(self.dust_var.get()) if hasattr(self, 'dust_var') else 0,
            int(self.min_fragment_var.get()) if hasattr(self, 'min_fragment_var') else 0,
            int(self.dust_var.get()) if hasattr(self, 'dust_var') else 0,
            
            # --- NEW: Add the clip value to the end of the payload (Index 18) ---
            int(self.bg_clip_var.get()) if hasattr(self, 'bg_clip_var') else 255,
        )
    # Record latest params; render or queue
        self._pending_params = params
        if not self._render_running:
            self._render_running = True
            threading.Thread(target=self._render_worker, args=(params,), daemon=True).start()
        else:
            # Already rendering -> queue
            self.set_status('Updating (queued)…')


    def _render_worker(self, params):
        try:
            # Check if shading parameters (first 6) have changed
            shading_params = params[:6]
            if self._cached_enh_img is not None and self._last_shading_params == shading_params:
                enh = self._cached_enh_img
            else:
                flat, enh, binary = pipeline_shading_sauvola(self.src_img, sigma=params[0], clip=params[1], tile=params[2], win=params[3], k=params[4], post_open=params[5])
                self._cached_enh_img = enh
                self._last_shading_params = shading_params
            
            # --- NEW: Apply the Background Clip ---
            enh = enh.copy() # Protect the cache!
            
            # Use [-1] to always grab the last item in the payload!
            bg_clip_val = params[-1] if len(params) > 18 else 255
            
            if bg_clip_val < 255:
                # This is the magic numpy trick: Find all pixels brighter than the clip, and force them to pure white.
                enh[enh > bg_clip_val] = 255
            # --------------------------------------

            self.enh_img = enh
            method = 'Otsu'
            bw = binarize(enh, method=method, blur=0)
            bw01 = (bw > 0).astype(np.uint8)
            
            # Use fixed 1° step for orientation sampling
            max_ang = float(self.k_ang.get()) if hasattr(self, 'k_ang') else 8.0
            calc_step = 1.0
            
            opened = oriented_opening(bw01, length=int(self.k_len.get()) if hasattr(self, 'k_len') else 41,
                                      thickness=int(self.k_thk.get()) if hasattr(self, 'k_thk') else 1,
                                      max_angle=max_ang,
                                      step=calc_step)
            if hasattr(self, 'k_dilate') and int(self.k_dilate.get()) > 0:
                K = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.k_dilate.get()), int(self.k_dilate.get())))
                opened = cv2.dilate(opened, K, 1)
            if hasattr(self, 'k_area') and int(self.k_area.get()) > 0:
                opened_bool = opened.astype(bool)
                opened_bool = remove_small_objects(opened_bool, min_size=int(self.k_area.get()))
                opened = opened_bool.astype(np.uint8)
            
            # Fill small loops (holes in binary mask)
            if len(params) > 15 and params[15] > 0:
                opened = fill_holes(opened, min_size=params[15])

            traced = skeletonize(opened.astype(bool)).astype(np.uint8)
            
            # Apply spur pruning if requested
            if len(params) > 14 and params[14] > 0:
                traced = remove_branches(traced, max_length=params[14])
            
            # Apply steep segment removal using Max angle
            max_ang = float(self.k_ang.get()) if hasattr(self, 'k_ang') else 0.0
            if max_ang > 0:
                 traced = remove_steep_segments(traced, max_angle_deg=max_ang)
            
            # Apply hump removal if requested
            if len(params) > 12 and params[12]:
                max_w = params[13] if len(params) > 13 else 2
                traced = remove_humps(traced, max_width=max_w)

            # Apply dust removal (small object removal on skeleton)
            if len(params) > 17 and params[17] > 0:
                traced_bool = traced.astype(bool)
                # Use connectivity=2 to handle 8-connected pixels in skeleton
                traced_bool = remove_small_objects(traced_bool, min_size=params[17], connectivity=2)
                traced = traced_bool.astype(np.uint8)

            # Normalize to 0/255 if traced is 0/1
            if traced.max() == 1:
                traced = traced * 255

            # Ensure traced is 0/255 before creating mask
            # If traced was already 0/255, the above elif didn't run.
            # If traced was 0/1, it is now 0/255.
            
            # Final check to ensure we are in 0/255 range for display/saving
            if traced.max() == 1:
                 traced = traced * 255

            self._binary_mask = 255 - traced
            # bg_fade is inverse of opacity: 0 => full gray, 1 => white
            bg_fade = 1.0 - (float(self.bg_opacity.get()) if hasattr(self, 'bg_opacity') else 1.0)
            overlay = overlay_mask_on_gray(enh, traced, line_alpha=1.0,
                                           bg_fade=bg_fade,
                                           bg_to='white')
            illum_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR) if enh.ndim == 2 else enh
            self.after(0, lambda: (self._update_illum_and_fringe(illum_bgr, overlay), self.update_idletasks()))
        except Exception:
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            def _maybe_run_next():
                try:
                    next_params = self._pending_params
                    if next_params is not None and next_params != params:
                        self._pending_params = None
                        threading.Thread(target=self._render_worker, args=(next_params,), daemon=True).start()
                    else:
                        self._render_running = False
                        self._pending_params = None
                except Exception:
                    self._render_running = False
                    self._pending_params = None
            self.after(0, _maybe_run_next)

    # Public API for editor integration
    def update_with_overlay(self, illum_bgr, overlay_bgr):
        self._update_illum_and_fringe(illum_bgr, overlay_bgr)

    def apply_editor_mask(self, mask):
        """Apply externally edited binary mask."""
        try:
            self._binary_mask = mask.copy() if hasattr(mask, 'copy') else mask
            if self.enh_img is not None:
                traced = (255 - self._binary_mask) // 255
                bg_fade = 1.0 - (float(self.bg_opacity.get()) if hasattr(self, 'bg_opacity') else 1.0)
                overlay = overlay_mask_on_gray(
                    self.enh_img,
                    traced.astype(np.uint8),
                    line_alpha=1.0,
                    bg_fade=bg_fade,
                    bg_to='white'
                )
                self._update_illum_and_fringe(cv2.cvtColor(self.enh_img, cv2.COLOR_GRAY2BGR), overlay)
                try:
                    self.update_idletasks()
                except Exception:
                    pass
                self.set_status('Applied edited mask from Editor')
        except Exception:
            pass

    # Rendering
    def _update_illum_and_fringe(self, illum_bgr, overlay_bgr):
        try:
            self._last_illum_bgr = illum_bgr
            self._last_overlay_bgr = overlay_bgr
            # Original overlay feature removed: display processed illumination and fringe directly.
            fringe_disp = overlay_bgr

            # Fringe viewer
            if fringe_disp is not None:
                photo_fringe = to_photoimage_from_bgr_with_scale(fringe_disp, scale=self._fringe_zoom)
                self._photo_fringe = photo_fringe
                if self._fringe_img_id is None:
                    self._fringe_img_id = self.fringe_canvas.create_image(0, 0, anchor='nw', image=photo_fringe)
                else:
                    self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                try:
                    cw = max(1, int(self.fringe_canvas.winfo_width()))
                    ch = max(1, int(self.fringe_canvas.winfo_height()))
                except Exception:
                    cw = photo_fringe.width(); ch = photo_fringe.height()
                iw, ih = photo_fringe.width(), photo_fringe.height()
                margin = max(cw, ch, iw, ih) // 2
                self.fringe_canvas.config(scrollregion=(-margin, -margin, iw + margin, ih + margin))
                if not self._fringe_centered:
                    total_w = (iw + 2 * margin); total_h = (ih + 2 * margin)
                    left_px = margin + max(0, (iw - cw) // 2)
                    top_px = margin + max(0, (ih - ch) // 2)
                    try:
                        self.fringe_canvas.xview_moveto(left_px / max(1, total_w))
                        self.fringe_canvas.yview_moveto(top_px / max(1, total_h))
                    except Exception:
                        pass
                    self._fringe_centered = True

            self._zoom_level = self._fringe_zoom
            self.set_status('Rendered')
        except Exception:
            pass
