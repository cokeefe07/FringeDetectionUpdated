import tkinter as tk
from tkinter import ttk

from tabs.overlay_tab import OverlayTabFrame
from tabs.detection_tab import DetectionTabFrame


class EvenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SpellBook — Fringe Detection')
        self.geometry('1100x700')
        try:
            self.after(100, self.lift)
            self.after(120, lambda: self.attributes('-topmost', True))
            self.after(700, lambda: self.attributes('-topmost', False))
        except Exception:
            pass

        # Notebook tabs: Overlay, Detection, Editor
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Optional status bar at bottom
        self.statusbar = ttk.Label(self, text='Ready')
        self.statusbar.pack(fill='x', side='bottom')

        # Overlay tab
        try:
            self._overlay_frame = OverlayTabFrame(self.notebook, status_callback=self.set_status)
            self.notebook.add(self._overlay_frame, text='Overlay')
        except Exception:
            self._overlay_frame = None

        # Detection tab
        try:
            self._detection_frame = DetectionTabFrame(self.notebook, status_callback=self.set_status)
            self.notebook.add(self._detection_frame, text='Detection')
        except Exception as e:
            self._detection_frame = None
            try:
                self.set_status('Failed to load Detection tab')
            except Exception:
                pass
            # Log exception to console for debugging
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

        # Editor tab
        editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(editor_tab, text='Editor')
        
        from tabs.fringe_editor import FringeEditorFrame
        #try:
           # from tabs.fringe_editor import FringeEditorFrame
        #except Exception:
        #    FringeEditorFrame = None
        self._editor_frame = None
        if FringeEditorFrame is not None:
            def on_apply(mask, _bg):
                try:
                    if self._detection_frame is not None:
                        self._detection_frame.apply_editor_mask(mask)
                        self.set_status('Applied edited mask from Editor')
                except Exception:
                    pass

            def on_close():
                pass

            self._editor_frame = FringeEditorFrame(editor_tab, on_apply=on_apply, on_close=on_close)
            self._editor_frame.pack(fill='both', expand=True)

    def set_status(self, txt):
        try:
            self.statusbar.config(text=txt)
        except Exception:
            pass

    def on_close(self):
        try:
            self.destroy()
        except Exception:
            pass


def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
