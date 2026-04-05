import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from fringe_detection import to_photoimage_from_bgr_with_scale

class OverlayTabFrame(ttk.Frame):
    """Full overlay implementation (reference anchored, shot movable, zoom/pan, crop, nudge, save)."""
    def __init__(self, master, status_callback=None):
        super().__init__(master)
        self._status = status_callback or (lambda txt: None)
        self._shot = None; self._ref = None
        self._shot_bgr = None; self._ref_bgr = None
        self._zoom = 1.0; self._base_scale = 1.0; self._scale = 1.0
        self._offset = None
        self._shot_dragging = False
        self._shot_drag_start = (0,0)
        self._shot_rel_offset = (0.0,0.0)
        self._shot_rel_offset_start = (0.0,0.0)
        self._panning = False
        self._pan_start = (0,0); self._offset_start = (0,0)
        self._crop_mode = False; self._crop_rect = None; self._crop_drag=None; self._crop_last=None; self._crop_items=[]; self._crop_handle_px=6
        self._render_job=None; self._photo=None; self._img_id=None; self._prev_ref_shift=None
        self._arrow_btns={}
        self._build_ui(); self._bind_events()

    def _build_ui(self):
        # Left control panel container
        ctrl_container = ttk.Frame(self)
        ctrl_container.pack(side='left', fill='y', padx=8, pady=8)
        ctrl_container.config(width=260)
        ctrl_container.pack_propagate(False)

        # Control frame (non-scrollable)
        try:
            bg_color = self.cget('background') or '#f0f0f0'
        except Exception:
            bg_color = '#f0f0f0'
        ctrl = ttk.Frame(ctrl_container)
        ctrl.pack(fill='both', expand=True)

        # Title + help icon
        title_row = ttk.Frame(ctrl); title_row.pack(anchor='w', fill='x')
        ttk.Label(title_row, text='Overlay', font=('Segoe UI', 10, 'bold')).pack(side='left')
        def make_help_icon(parent, tooltip_text):
            # Canvas-based thin circle with non-bold question mark
            try:
                bg = self.cget('background')
            except Exception:
                bg = '#f0f0f0'
            c = tk.Canvas(parent, width=18, height=18, highlightthickness=0, bg=bg)
            c.create_oval(2, 2, 16, 16, outline='#666', width=1)
            c.create_text(9, 9, text='?', font=('Segoe UI', 9))
            c.pack(side='left', padx=(6,0))
            self._attach_tooltip(c, tooltip_text)
            return c
        make_help_icon(title_row, (
            'Overlay Tab Purpose\n'
            'Used to align a Reference and Shot image then crop both\n'
            'to the same dimensions\n'
            '\n'
            'Controls:\n'
            '- Right-click drag to move both images\n'
            '- Left-click drag to move Shot image\n'
            '- Mouse wheel to zoom\n'
            '- Arrow keys for 1px adjustment of Shot image\n'
            '\n'
            'Features:\n'
            '- Load Reference and Shot images from RawImages\n'
            '- Shot opacity: Change opacity of Shot image\n'
            '- Brightness sliders: Adjust brightness of each image\n'
            '- Crop mode: Define and apply crop to both images\n'
            '- Nudge buttons for 1px adjustment of Shot image\n'
            '- Save Reference and Shot images to EditedImages\n'
        ))
        ref_row = ttk.Frame(ctrl)
        ref_row.pack(anchor='w', pady=4, fill='x')
        ttk.Button(ref_row, text='Load Reference Image', command=self._load_ref).pack(side='left')
        # moved: place Load Shot next to Load Reference
        ttk.Button(ref_row, text='Load Shot Image', command=self._load_shot).pack(side='left', padx=(6, 0))

        shot_row = ttk.Frame(ctrl)
        shot_row.pack(anchor='w', pady=4, fill='x')
        # moved: place Save Reference in the shot row next to Save Shot
        ttk.Button(shot_row, text='Save Reference Image', command=self._save_ref).pack(side='left')
        ttk.Button(shot_row, text='Save Shot Image', command=self._save_shot).pack(side='left', padx=(6, 0))
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        self.shot_alpha = tk.DoubleVar(value=0.5)
        shot_scale = self._make_slider_row(ctrl,'Shot opacity',self.shot_alpha,0.0,1.0,fmt='{:.2f}',command=lambda *_: self._schedule_render())
        # Make trough clicks step by ±0.1 instead of jumping to min/max
        def _shot_trough_click(event, scale=shot_scale, var=self.shot_alpha):
            try:
                w = max(1, int(scale.winfo_width()))
                x = int(getattr(event, 'x', 0))
                frm, to = 0.0, 1.0
                cur = float(var.get())
                rng = max(1e-9, to - frm)
                handle_x = int(round((cur - frm) / rng * w))
                # If clicking near the handle, allow default drag behavior
                if abs(x - handle_x) <= 10:
                    return None
                step = 0.1
                new = cur + (step if x > handle_x else -step)
                # Clamp and round to 2 decimals
                new = max(frm, min(to, round(new, 2)))
                var.set(new)
                self._schedule_render()
                return 'break'
            except Exception:
                return None
        try:
            shot_scale.bind('<Button-1>', _shot_trough_click)
        except Exception:
            pass
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        self.ref_brightness = tk.DoubleVar(value=8.0); self.shot_brightness = tk.DoubleVar(value=8.0)
        self._make_slider_row(ctrl,'Reference brightness',self.ref_brightness,1.0,10.0,fmt='{:.1f}',command=lambda *_: self._schedule_render())
        self._make_slider_row(ctrl,'Shot brightness',self.shot_brightness,1.0,10.0,fmt='{:.1f}',command=lambda *_: self._schedule_render())
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        crop_row = ttk.Frame(ctrl); crop_row.pack(fill='x', pady=4)
        ttk.Button(crop_row,text='Crop',command=self._toggle_crop_mode).pack(side='left', padx=(0,6))
        ttk.Button(crop_row,text='Apply',command=self._apply_crop).pack(side='left')
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        arrows = ttk.LabelFrame(ctrl,text='Nudge (1px)'); arrows.pack(fill='x', pady=4)
        grid = ttk.Frame(arrows); grid.pack(pady=4)
        def mk_btn(txt,ks,cmd,r,c):
            b = ttk.Button(grid,text=txt,width=4,command=lambda:(self._indicate_arrow_press(ks,True,True),cmd()))
            b.grid(row=r,column=c,padx=2,pady=2); self._arrow_btns[ks]=b
        mk_btn('↑','Up',lambda: self._nudge_shot(0,-1),0,1)
        mk_btn('←','Left',lambda: self._nudge_shot(-1,0),1,0)
        mk_btn('→','Right',lambda: self._nudge_shot(1,0),1,2)
        mk_btn('↓','Down',lambda: self._nudge_shot(0,1),2,1)
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        # Save buttons moved next to their corresponding Load buttons above
        right = ttk.Frame(self); right.pack(side='left', fill='both', expand=True, padx=8, pady=8); right.pack_propagate(False)
        self.canvas = tk.Canvas(right,bg='black', highlightthickness=0); self.canvas.pack(fill='both', expand=True)

    def _attach_tooltip(self, widget, text):
        tip = {'win': None}
        def show_tip(_e=None):
            if tip['win'] is not None: return
            try:
                x = widget.winfo_rootx() + widget.winfo_width() + 8
                y = widget.winfo_rooty() + int(widget.winfo_height()*0.5)
            except Exception:
                x = y = 0
            win = tk.Toplevel(widget); tip['win'] = win
            try: win.wm_overrideredirect(True)
            except Exception: pass
            try: win.wm_geometry(f"+{x}+{y}")
            except Exception: pass
            frame = ttk.Frame(win, borderwidth=1, relief='solid'); frame.pack()
            lbl = ttk.Label(frame, text=text, justify='left', padding=6); lbl.pack()
        def hide_tip(_e=None):
            w = tip.get('win')
            if w is not None:
                try: w.destroy()
                except Exception: pass
                tip['win'] = None
        try:
            widget.bind('<Enter>', show_tip); widget.bind('<Leave>', hide_tip)
        except Exception:
            pass

    def _bind_events(self):
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-3>', self._on_pan_start); self.canvas.bind('<B3-Motion>', self._on_pan_move); self.canvas.bind('<ButtonRelease-3>', self._on_pan_end)
        self.canvas.bind('<Button-1>', self._on_shot_drag_start); self.canvas.bind('<B1-Motion>', self._on_shot_drag_move); self.canvas.bind('<ButtonRelease-1>', self._on_shot_drag_end)
        self.canvas.bind('<Enter>', lambda e: self.canvas.focus_set())
        for key in ('Left','Right','Up','Down'):
            self.canvas.bind(f'<KeyPress-{key}>', self._on_key_nudge)
            self.canvas.bind(f'<KeyRelease-{key}>', self._on_key_release)

    def _make_slider_row(self,parent,label_text,var,frm,to,fmt='{:.2f}',command=None):
        row=ttk.Frame(parent); row.pack(fill='x', pady=4)
        ttk.Label(row,text=label_text).pack(side='left')
        val_lbl=ttk.Label(row,width=6,anchor='e'); val_lbl.pack(side='right')
        def on_slide(val=None):
            try: v=float(val) if val is not None else float(var.get()); var.set(v)
            except Exception: v=0.0
            val_lbl.config(text=fmt.format(v));
            if command: command(v)
        scale=ttk.Scale(row,from_=frm,to=to,orient='horizontal',command=on_slide); scale.pack(side='left', fill='x', expand=True, padx=8)
        try: scale.configure(value=float(var.get()))
        except Exception: pass
        def on_var_change(*_):
            try: scale.configure(value=float(var.get()))
            except Exception: pass
            on_slide()
        var.trace_add('write', on_var_change); on_slide(); return scale

    def _load_ref(self):
        # Open in RawImages directory if it exists
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        raw_dir = os.path.join(base_dir, 'RawImages')
        initial = raw_dir if os.path.isdir(raw_dir) else base_dir
        p=filedialog.askopenfilename(parent=self,title='Select reference image',initialdir=initial,filetypes=[('Images',('*.png','*.jpg','*.jpeg','*.tif','*.tiff'))])
        if not p: return
        img=cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None: messagebox.showerror('Load error','Failed to read reference image'); return
        self._ref=self._normalize_image_dtype(img); self._prepare_sources(); self._status(f'Loaded reference {os.path.basename(p)}'); self._schedule_render()

    def _load_shot(self):
        # Open in RawImages directory if it exists
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        raw_dir = os.path.join(base_dir, 'RawImages')
        initial = raw_dir if os.path.isdir(raw_dir) else base_dir
        p=filedialog.askopenfilename(parent=self,title='Select shot image',initialdir=initial,filetypes=[('Images',('*.png','*.jpg','*.jpeg','*.tif','*.tiff'))])
        if not p: return
        img=cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None: messagebox.showerror('Load error','Failed to read shot image'); return
        self._shot=self._normalize_image_dtype(img); self._prepare_sources();
        try: self.shot_alpha.set(0.5)
        except Exception: pass
        self._status(f'Loaded shot {os.path.basename(p)}'); self._schedule_render()

    def _normalize_image_dtype(self,img):
        if img is None: return None
        if img.dtype==np.uint8: return img
        try:
            norm=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
            if norm.dtype!=np.uint8: norm=norm.astype(np.uint8)
            return norm
        except Exception:
            mx=float(np.max(img)) if img.size else 1.0
            if mx<=0: return np.zeros_like(img,dtype=np.uint8)
            return (img.astype(np.float32)*(255.0/mx)).clip(0,255).astype(np.uint8)

    def _ensure_bgr(self,img):
        if img is None: return None
        try:
            if img.ndim==2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.ndim==3:
                if img.shape[2]==3: return img
                if img.shape[2]==4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception: pass
        try:
            g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        except Exception:
            return np.zeros((1,1,3),dtype=np.uint8)

    def _prepare_sources(self):
        self._shot_bgr=self._ensure_bgr(self._shot); self._ref_bgr=self._ensure_bgr(self._ref)
        self._zoom=1.0; self._base_scale=1.0; self._offset=None; self._shot_rel_offset=(0.0,0.0); self._prev_ref_shift=None

    def _compute_world_bbox(self):
        sx_rel,sy_rel=self._shot_rel_offset
        refW=self._ref_bgr.shape[1] if self._ref_bgr is not None else 0; refH=self._ref_bgr.shape[0] if self._ref_bgr is not None else 0
        shotW=self._shot_bgr.shape[1] if self._shot_bgr is not None else 0; shotH=self._shot_bgr.shape[0] if self._shot_bgr is not None else 0
        x_min=0.0; y_min=0.0; x_max=float(refW); y_max=float(refH)
        if self._shot_bgr is not None:
            x_min=min(x_min,float(sx_rel)); y_min=min(y_min,float(sy_rel)); x_max=max(x_max,float(sx_rel+shotW)); y_max=max(y_max,float(sy_rel+shotH))
        if self._ref_bgr is None:
            x_min=min(0.0,float(sx_rel)); y_min=min(0.0,float(sy_rel)); x_max=max(0.0,float(sx_rel+shotW)); y_max=max(0.0,float(sy_rel+shotH))
        import math
        W=max(1,int(math.ceil(x_max-x_min))); H=max(1,int(math.ceil(y_max-y_min)))
        ref_shift=(-x_min,-y_min); shot_shift=(float(sx_rel)-x_min,float(sy_rel)-y_min)
        return x_min,y_min,W,H,ref_shift,shot_shift

    def _schedule_render(self,delay_ms=16):
        if self._render_job is not None:
            try: self.after_cancel(self._render_job)
            except Exception: pass
        self._render_job=self.after(delay_ms,self._render)

    def _on_canvas_configure(self,_e=None): self._schedule_render()

    def _render(self):
        if self._shot_bgr is None and self._ref_bgr is None: self.canvas.delete('all'); return
        shot=self._shot_bgr; ref=self._ref_bgr
        x_min,y_min,W,H,ref_shift,shot_shift=self._compute_world_bbox()
        try: sa=float(self.shot_alpha.get())
        except Exception: sa=0.0; sa=max(0.0,min(1.0,sa))
        try: rb=float(self.ref_brightness.get())
        except Exception: rb=5.0
        try: sb=float(self.shot_brightness.get())
        except Exception: sb=5.0
        rb=max(1.0,min(10.0,rb)); sb=max(1.0,min(10.0,sb))
        cw=max(1,self.canvas.winfo_width()); ch=max(1,self.canvas.winfo_height())
        if self._offset is None or self._base_scale<=0: self._base_scale=max(1e-6,min(cw/float(W),ch/float(H)))
        self._scale=self._base_scale*self._zoom
        disp_w=max(1,int(round(W*self._scale))); disp_h=max(1,int(round(H*self._scale)))
        if self._offset is None: self._offset=self._centered_offset(W,H,self._scale)
        if ref is not None and self._shot_dragging:
            prev=self._prev_ref_shift
            if prev is not None:
                dxs=float(prev[0])-float(ref_shift[0]); dys=float(prev[1])-float(ref_shift[1])
                if abs(dxs)>1e-6 or abs(dys)>1e-6:
                    ox,oy=self._offset; self._offset=(int(round(ox+dxs*self._scale)), int(round(oy+dys*self._scale)))
        self._prev_ref_shift=ref_shift
        x0,y0=self._offset; vx0=max(0,x0); vy0=max(0,y0); vx1=min(cw,x0+disp_w); vy1=min(ch,y0+disp_h)
        self.canvas.delete('all')
        if vx1<=vx0 or vy1<=vy0: return
        import math
        sc=self._scale if self._scale>0 else 1.0
        ix0=int(max(0,math.floor((vx0-x0)/sc))); iy0=int(max(0,math.floor((vy0-y0)/sc)))
        ix1=int(min(W,math.ceil((vx1-x0)/sc))); iy1=int(min(H,math.ceil((vy1-y0)/sc)))
        if ix1<=ix0 or iy1<=iy0: return
        dst_w=vx1-vx0; dst_h=vy1-vy0
        fast=self._panning or self._shot_dragging
        interp=cv2.INTER_NEAREST if fast else cv2.INTER_LINEAR
        def extract(img,shift):
            if img is None: return np.zeros((iy1-iy0,ix1-ix0,3),dtype=np.uint8), np.zeros((iy1-iy0,ix1-ix0),dtype=np.uint8)
            return self._extract_shifted_region(img,ix0,iy0,ix1,iy1,shift[0],shift[1],img.shape[1],img.shape[0])
        ref_region,_=extract(ref,ref_shift); shot_region,shot_mask=extract(shot,shot_shift)
        try: ref_res=cv2.resize(ref_region,(dst_w,dst_h),interpolation=interp)
        except Exception: ref_res=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
        try: shot_res=cv2.resize(shot_region,(dst_w,dst_h),interpolation=interp); mask_res=cv2.resize(shot_mask,(dst_w,dst_h),interpolation=cv2.INTER_NEAREST).astype(np.float32)
        except Exception: shot_res=np.zeros((dst_h,dst_w,3),dtype=np.uint8); mask_res=np.zeros((dst_h,dst_w),dtype=np.float32)
        ref_scaled=np.clip(ref_res.astype(np.float32)*rb,0,255); shot_scaled=np.clip(shot_res.astype(np.float32)*sb,0,255)
        comp=ref_scaled*(1.0-sa*mask_res[:,:,None]) + shot_scaled*(sa*mask_res[:,:,None]); comp=np.clip(comp,0,255).astype(np.uint8)
        if self._crop_mode and self._crop_rect is not None and ref is not None: comp=self._shade_crop(comp,ref_shift,(vx0,vy0),(ix0,iy0),sc)
        try: photo=to_photoimage_from_bgr_with_scale(comp,scale=1.0)
        except Exception: im=Image.fromarray(cv2.cvtColor(comp,cv2.COLOR_BGR2RGB)); photo=ImageTk.PhotoImage(im)
        self._photo=photo; self._img_id=self.canvas.create_image(vx0,vy0,anchor='nw',image=photo); self._redraw_crop_overlay()

    def _shade_crop(self,comp,ref_shift,view_origin,crop_ix_origin,scale):
        # Compute crop rectangle strictly in comp (image) coordinates so it stays aligned when panning/zooming.
        x,y,w,h=self._crop_rect; rx,ry=ref_shift
        # Top-left of the visible comp (in world index space)
        ix0, iy0 = crop_ix_origin
        # Convert world-space crop rect -> comp-space pixels; don't add view_origin (canvas offset)
        cx0=int(round((rx + x     - ix0)*scale)); cy0=int(round((ry + y     - iy0)*scale))
        cx1=int(round((rx + x + w - ix0)*scale)); cy1=int(round((ry + y + h - iy0)*scale))
        H,W=comp.shape[:2]
        # Clamp to comp bounds
        cx0=max(0,min(W,cx0)); cy0=max(0,min(H,cy0)); cx1=max(0,min(W,cx1)); cy1=max(0,min(H,cy1))
        # Build dimming mask: dim outside the crop, keep crop area undimmed
        mask=np.zeros((H,W),dtype=np.uint8); mask[:]=70
        mask[cy0:cy1,cx0:cx1]=0
        return (comp.astype(np.int16)-mask[...,None]).clip(0,255).astype(np.uint8)

    def _centered_offset(self,W,H,scale):
        cw=max(1,self.canvas.winfo_width()); ch=max(1,self.canvas.winfo_height())
        disp_w=int(round(W*scale)); disp_h=int(round(H*scale)); return ((cw-disp_w)//2,(ch-disp_h)//2)

    def _extract_shifted_region(self,img,ix0,iy0,ix1,iy1,sx,sy,W,H):
        dx=int(round(sx)); dy=int(round(sy)); h=max(0,iy1-iy0); w=max(0,ix1-ix0)
        if h==0 or w==0: return np.zeros((h,w,3),dtype=np.uint8), np.zeros((h,w),dtype=np.uint8)
        src_x0=ix0-dx; src_y0=iy0-dy; src_x1=ix1-dx; src_y1=iy1-dy
        clip_x0=max(0,src_x0); clip_y0=max(0,src_y0); clip_x1=min(W,src_x1); clip_y1=min(H,src_y1)
        if clip_x1<=clip_x0 or clip_y1<=clip_y0: return np.zeros((h,w,3),dtype=np.uint8), np.zeros((h,w),dtype=np.uint8)
        dst_x0=clip_x0-src_x0; dst_y0=clip_y0-src_y0; overl_w=clip_x1-clip_x0; overl_h=clip_y1-clip_y0
        out=np.zeros((h,w,3),dtype=np.uint8); mask=np.zeros((h,w),dtype=np.uint8)
        out[dst_y0:dst_y0+overl_h,dst_x0:dst_x0+overl_w]=img[clip_y0:clip_y0+overl_h,clip_x0:clip_x0+overl_w]
        mask[dst_y0:dst_y0+overl_h,dst_x0:dst_x0+overl_w]=1; return out,mask

    def _on_mouse_wheel(self,event):
        if self._shot_bgr is None and self._ref_bgr is None: return
        delta=1 if event.delta>0 else -1; zoom_factor=1.1 if delta>0 else 0.9
        old_zoom=self._zoom; new_zoom=max(0.1,min(64.0,old_zoom*zoom_factor))
        if abs(new_zoom-old_zoom)<1e-6: return
        x_c,y_c=event.x,event.y; _,_,W,H,_,_=self._compute_world_bbox(); cw=max(1,self.canvas.winfo_width()); ch=max(1,self.canvas.winfo_height())
        if self._offset is None or self._base_scale<=0: self._base_scale=max(1e-6,min(cw/float(W),ch/float(H)))
        s_before=max(1e-6,self._base_scale*old_zoom); s_after=max(1e-6,self._base_scale*new_zoom)
        ox,oy=self._offset if self._offset is not None else self._centered_offset(W,H,s_before)
        x_img=(x_c-ox)/s_before; y_img=(y_c-oy)/s_before
        self._zoom=new_zoom; new_ox=int(round(x_c-x_img*s_after)); new_oy=int(round(y_c-y_img*s_after)); self._offset=(new_ox,new_oy); self._schedule_render()

    def _on_pan_start(self,event):
        self._panning=True; self._pan_start=(event.x,event.y); self._offset_start=self._offset if self._offset is not None else (0,0)
        try: self.config(cursor='fleur')
        except Exception: pass

    def _on_pan_move(self,event):
        if not self._panning: return
        sx,sy=self._pan_start; dx=event.x-sx; dy=event.y-sy; ox0,oy0=self._offset_start; self._offset=(ox0+dx,oy0+dy); self._render()

    def _on_pan_end(self,_e):
        self._panning=False
        try: self.config(cursor='')
        except Exception: pass
        self._render()

    def _on_shot_drag_start(self,event):
        if self._crop_mode: self._on_crop_press(event); return 'break'
        if self._shot_bgr is None: return
        self._shot_dragging=True; self._shot_drag_start=(event.x,event.y); self._shot_rel_offset_start=self._shot_rel_offset
        try: self.config(cursor='hand2')
        except Exception: pass

    def _on_shot_drag_move(self,event):
        if self._crop_mode: self._on_crop_drag(event); return 'break'
        if not self._shot_dragging: return
        sx,sy=self._shot_drag_start; dx=event.x-sx; dy=event.y-sy; s=self._scale if self._scale>1e-6 else 1.0; ox0,oy0=self._shot_rel_offset_start
        self._shot_rel_offset=(ox0+dx/s, oy0+dy/s); self._render()

    def _on_shot_drag_end(self,event):
        if self._crop_mode: self._on_crop_release(event); return 'break'
        if not self._shot_dragging: return
        self._shot_dragging=False
        try: self.config(cursor='')
        except Exception: pass
        self._render()

    def _indicate_arrow_press(self,keysym,pressed=True,auto_release=False):
        btn=self._arrow_btns.get(keysym)
        if not btn: return
        if pressed:
            btn.state(['pressed'])
            if auto_release: self.after(120, lambda: self._indicate_arrow_press(keysym,False,False))
        else:
            btn.state(['!pressed'])

    def _on_key_nudge(self,event):
        ks=getattr(event,'keysym','')
        if ks not in ('Left','Right','Up','Down'): return
        self._indicate_arrow_press(ks,True,False)
        if ks=='Left': self._nudge_shot(-1,0)
        elif ks=='Right': self._nudge_shot(1,0)
        elif ks=='Up': self._nudge_shot(0,-1)
        elif ks=='Down': self._nudge_shot(0,1)

    def _on_key_release(self,event):
        ks=getattr(event,'keysym','')
        if ks in ('Left','Right','Up','Down'): self._indicate_arrow_press(ks,False,False)

    def _nudge_shot(self,dx,dy):
        if self._shot_bgr is None: return
        try: _,_,_,_, ref_before,_=self._compute_world_bbox()
        except Exception: ref_before=(0.0,0.0)
        sx,sy=self._shot_rel_offset; self._shot_rel_offset=(sx+dx, sy+dy)
        try: _,_,_,_, ref_after,_=self._compute_world_bbox()
        except Exception: ref_after=ref_before
        try:
            s=self._scale if self._scale>1e-6 else (self._base_scale if self._base_scale>1e-6 else 1.0)
            dox=(ref_before[0]-ref_after[0])*s; doy=(ref_before[1]-ref_after[1])*s
            if self._offset is None:
                _,_,W,H,_,_=self._compute_world_bbox(); self._offset=self._centered_offset(W,H,s)
            ox,oy=self._offset; self._offset=(int(round(ox+dox)), int(round(oy+doy)))
        except Exception: pass
        self._render()

    def _save_ref(self):
        if self._ref_bgr is None: messagebox.showinfo('Save Reference','No reference image loaded.'); return
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        edited_dir = os.path.join(base_dir, 'EditedImages')
        os.makedirs(edited_dir, exist_ok=True)
        p=filedialog.asksaveasfilename(parent=self,title='Save reference image',initialdir=edited_dir,defaultextension='.png',filetypes=[('PNG','*.png'),('JPEG','*.jpg'),('TIFF','*.tif'),('All','*.*')])
        if not p: return
        try: cv2.imwrite(p,self._ref_bgr); self._status(f'Saved reference to {os.path.basename(p)}')
        except Exception: messagebox.showerror('Save error','Failed to save reference image')

    def _save_shot(self):
        if self._shot_bgr is None: messagebox.showinfo('Save Shot','No shot image loaded.'); return
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        edited_dir = os.path.join(base_dir, 'EditedImages')
        os.makedirs(edited_dir, exist_ok=True)
        p=filedialog.asksaveasfilename(parent=self,title='Save shot image',initialdir=edited_dir,defaultextension='.png',filetypes=[('PNG','*.png'),('JPEG','*.jpg'),('TIFF','*.tif'),('All','*.*')])
        if not p: return
        try: cv2.imwrite(p,self._shot_bgr); self._status(f'Saved shot to {os.path.basename(p)}')
        except Exception: messagebox.showerror('Save error','Failed to save shot image')

    def _toggle_crop_mode(self):
        if not self._crop_mode:
            H,W=self._get_world_hw(); x=max(0,W//4); y=max(0,H//4); w=max(1,W//2); h=max(1,H//2)
            self._crop_rect=[float(x),float(y),float(w),float(h)]; self._crop_mode=True; self._redraw_crop_overlay()
        else:
            self._crop_mode=False; self._crop_rect=None; self._crop_drag=None; self._crop_last=None; self._clear_crop_overlay(); self._schedule_render()

    def _get_world_hw(self):
        x_min,y_min,W,H,_,_=self._compute_world_bbox(); return H,W

    def _clear_crop_overlay(self):
        if not self._crop_items: return
        for it in self._crop_items:
            try: self.canvas.delete(it)
            except Exception: pass
        self._crop_items=[]

    def _redraw_crop_overlay(self):
        self._clear_crop_overlay();
        if not self._crop_mode or self._crop_rect is None: return
        try:
            x,y,w,h=self._crop_rect; _,_,W,H,ref_shift,_=self._compute_world_bbox(); rx,ry=ref_shift; ox,oy=self._offset if self._offset is not None else (0,0)
            s=self._scale if self._scale>1e-6 else 1.0
            cx0=int(round(ox+(rx+x)*s)); cy0=int(round(oy+(ry+y)*s)); cx1=int(round(ox+(rx+x+w)*s)); cy1=int(round(oy+(ry+y+h)*s))
            rect_id=self.canvas.create_rectangle(cx0,cy0,cx1,cy1,outline='#ffd400',width=2,dash=(4,3)); self._crop_items.append(rect_id)
            r=self._crop_handle_px
            for (hx,hy) in [(cx0,cy0),(cx1,cy0),(cx0,cy1),(cx1,cy1)]:
                try: hid=self.canvas.create_rectangle(hx-r,hy-r,hx+r,hy+r,fill='#ffd400',outline=''); self._crop_items.append(hid)
                except Exception: pass
            try: tid=self.canvas.create_text(cx0+6,cy0-10,anchor='w',text=f'({int(x)},{int(y)}) {int(w)}×{int(h)}',fill='#ffd400'); self._crop_items.append(tid)
            except Exception: pass
        except Exception: pass

    def _on_crop_press(self,event):
        if not self._crop_mode or self._crop_rect is None: return
        x,y,w,h=self._crop_rect; _,_,W,H,ref_shift,_=self._compute_world_bbox(); rx,ry=ref_shift; ox,oy=self._offset if self._offset is not None else (0,0); s=self._scale if self._scale>1e-6 else 1.0
        cx0=ox+(rx+x)*s; cy0=oy+(ry+y)*s; cx1=ox+(rx+x+w)*s; cy1=oy+(ry+y+h)*s; ex,ey=float(event.x),float(event.y); r=float(self._crop_handle_px)*1.8
        def near(ax,ay): return abs(ex-ax)<=r and abs(ey-ay)<=r
        if near(cx0,cy0): self._crop_drag='tl'
        elif near(cx1,cy0): self._crop_drag='tr'
        elif near(cx0,cy1): self._crop_drag='bl'
        elif near(cx1,cy1): self._crop_drag='br'
        elif ex>=cx0 and ex<=cx1 and ey>=cy0 and ey<=cy1: self._crop_drag='move'
        else: self._crop_drag=None
        self._crop_last=((ex-ox)/s - rx, (ey-oy)/s - ry)

    def _on_crop_drag(self,event):
        if not self._crop_mode or self._crop_rect is None or self._crop_drag is None: return
        x,y,w,h=self._crop_rect; _,_,W,H,ref_shift,_=self._compute_world_bbox(); rx,ry=ref_shift; ox,oy=self._offset if self._offset is not None else (0,0); s=self._scale if self._scale>1e-6 else 1.0
        xi=(event.x-ox)/s - rx; yi=(event.y-oy)/s - ry; last=self._crop_last
        if last is None: self._crop_last=(xi,yi); return
        dx=xi-last[0]; dy=yi-last[1]
        if self._crop_drag=='move': x+=dx; y+=dy
        elif self._crop_drag=='tl': x+=dx; y+=dy; w-=dx; h-=dy
        elif self._crop_drag=='tr': y+=dy; w+=dx; h-=dy
        elif self._crop_drag=='bl': x+=dx; w-=dx; h+=dy
        elif self._crop_drag=='br': w+=dx; h+=dy
        x=max(0.0,x); y=max(0.0,y); w=max(1.0,w); h=max(1.0,h)
        worldH,worldW=self._get_world_hw()
        if x+w>worldW:
            if self._crop_drag=='move': x=worldW-w
            else: w=worldW-x
        if y+h>worldH:
            if self._crop_drag=='move': y=worldH-h
            else: h=worldH-y
        self._crop_rect=[x,y,w,h]; self._crop_last=(xi,yi); self._redraw_crop_overlay()

    def _on_crop_release(self,_e): self._crop_drag=None; self._crop_last=None

    def _apply_crop(self):
        if not self._crop_mode or self._crop_rect is None: return
        x,y,w,h=self._crop_rect; x1=x+w; y1=y+h
        def crop_img(img,shift):
            if img is None: return None
            sx,sy=shift; X0=int(max(0,round(x-sx))); Y0=int(max(0,round(y-sy))); X1=int(max(X0+1,round(x1-sx))); Y1=int(max(Y0+1,round(y1-sy)))
            hI, wI = img.shape[:2]
            X1 = min(wI, X1)
            Y1 = min(hI, Y1)
            if X1 <= X0 or Y1 <= Y0:
                return None
            return img[Y0:Y1,X0:X1].copy()
        _,_,_,_, ref_shift, shot_shift=self._compute_world_bbox(); self._ref=crop_img(self._ref,ref_shift); self._shot=crop_img(self._shot,shot_shift)
        self._prepare_sources(); self._toggle_crop_mode(); self._render()

# End of file

