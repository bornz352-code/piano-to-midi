"""Piano → MIDI — main GUI application."""
from __future__ import annotations
import sys
import os
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure project root is on the path when running from gui/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.loader import load as load_audio
from audio.recorder import Recorder
from pitch.detector import detect, NoteEvent
from midi.builder import build as build_midi, save as save_midi

ACCENT = "#4a9eff"
BG = "#1e1e2e"
FG = "#cdd6f4"
PANEL = "#313244"
NOTE_COLORS = {
    "bass":    "#89b4fa",  # blue  — MIDI 21-47
    "mid":     "#a6e3a1",  # green — MIDI 48-71
    "treble":  "#fab387",  # peach — MIDI 72-108
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Piano → MIDI")
        self.geometry("680x560")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._recorder = Recorder()
        self._recording = False
        self._events: list[NoteEvent] = []
        self._midi_file = None
        self._audio_duration = 0.0

        self._apply_style()
        self._build()

    # ── Style ─────────────────────────────────────────────────────────────────
    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                        insertcolor=FG, troughcolor=PANEL, borderwidth=0)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TButton", background=PANEL, foreground=FG, padding=(10, 5))
        style.map("TButton", background=[("active", "#45475a")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#11111b", padding=(10, 5))
        style.map("Accent.TButton", background=[("active", "#74c7ec")])
        style.configure("TNotebook", background=BG, tabmargins=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=FG, padding=(12, 5))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "#11111b")])
        style.configure("TSpinbox", background=PANEL, foreground=FG, arrowcolor=FG)
        style.configure("TEntry", background=PANEL, foreground=FG)
        style.configure("TProgressbar", troughcolor=PANEL, background=ACCENT)
        style.configure("TSeparator", background="#45475a")

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        hdr = ttk.Frame(self, padding=(20, 14, 20, 0))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Piano → MIDI Converter", font=("Helvetica", 17, "bold")).pack(side="left")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", padx=20, pady=10)

        file_tab = ttk.Frame(nb, padding=16)
        nb.add(file_tab, text="  Convert File  ")
        self._build_file_tab(file_tab)

        rec_tab = ttk.Frame(nb, padding=16)
        nb.add(rec_tab, text="  Record & Convert  ")
        self._build_rec_tab(rec_tab)

        # Piano roll preview
        roll_frame = ttk.Frame(self, padding=(20, 0, 20, 0))
        roll_frame.pack(fill="x")
        ttk.Label(roll_frame, text="Note Preview", font=("Helvetica", 10)).pack(anchor="w")
        self._roll = tk.Canvas(roll_frame, height=100, bg=PANEL, highlightthickness=0)
        self._roll.pack(fill="x", pady=(4, 8))

        # Status bar
        bot = ttk.Frame(self, padding=(20, 0, 20, 10))
        bot.pack(fill="x")
        self._status = tk.StringVar(value="Ready.")
        ttk.Label(bot, textvariable=self._status, foreground="#6c7086").pack(anchor="w")
        self._bar = ttk.Progressbar(bot, maximum=100, length=640)
        self._bar.pack(fill="x", pady=(4, 0))

    def _build_file_tab(self, f: ttk.Frame):
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Input audio:").grid(row=0, column=0, sticky="w", pady=6)
        self._in_path = tk.StringVar()
        ttk.Entry(f, textvariable=self._in_path, width=48).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(f, text="Browse…", command=self._browse_in).grid(row=0, column=2)

        ttk.Label(f, text="Output MIDI:").grid(row=1, column=0, sticky="w", pady=6)
        self._out_path = tk.StringVar()
        ttk.Entry(f, textvariable=self._out_path, width=48).grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(f, text="Browse…", command=self._browse_out).grid(row=1, column=2)

        ttk.Label(f, text="Tempo (BPM):").grid(row=2, column=0, sticky="w", pady=6)
        self._tempo = tk.IntVar(value=120)
        ttk.Spinbox(f, from_=20, to=300, width=7, textvariable=self._tempo).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Button(f, text="Convert to MIDI", style="Accent.TButton", command=self._run_convert).grid(
            row=3, column=0, columnspan=3, pady=14,
        )

    def _build_rec_tab(self, f: ttk.Frame):
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Max duration (s):").grid(row=0, column=0, sticky="w", pady=6)
        self._rec_dur = tk.IntVar(value=15)
        ttk.Spinbox(f, from_=1, to=600, width=7, textvariable=self._rec_dur).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(f, text="Output MIDI:").grid(row=1, column=0, sticky="w", pady=6)
        self._rec_out = tk.StringVar()
        ttk.Entry(f, textvariable=self._rec_out, width=48).grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(f, text="Browse…", command=self._browse_rec_out).grid(row=1, column=2)

        ttk.Label(f, text="Tempo (BPM):").grid(row=2, column=0, sticky="w", pady=6)
        self._rec_tempo = tk.IntVar(value=120)
        ttk.Spinbox(f, from_=20, to=300, width=7, textvariable=self._rec_tempo).grid(row=2, column=1, sticky="w", padx=6)

        self._rec_btn = ttk.Button(f, text="● Start Recording", style="Accent.TButton", command=self._toggle_record)
        self._rec_btn.grid(row=3, column=0, columnspan=3, pady=14)

    # ── File dialogs ──────────────────────────────────────────────────────────
    def _browse_in(self):
        p = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.aiff *.m4a"), ("All", "*.*")],
        )
        if p:
            self._in_path.set(p)
            if not self._out_path.get():
                from pathlib import Path
                self._out_path.set(str(Path(p).with_suffix(".mid")))

    def _browse_out(self):
        p = filedialog.asksaveasfilename(
            title="Save MIDI",
            defaultextension=".mid",
            filetypes=[("MIDI", "*.mid"), ("All", "*.*")],
        )
        if p:
            self._out_path.set(p)

    def _browse_rec_out(self):
        p = filedialog.asksaveasfilename(
            title="Save MIDI",
            defaultextension=".mid",
            filetypes=[("MIDI", "*.mid"), ("All", "*.*")],
        )
        if p:
            self._rec_out.set(p)

    # ── Conversion pipeline ───────────────────────────────────────────────────
    def _run_convert(self):
        inp = self._in_path.get().strip()
        out = self._out_path.get().strip()
        if not inp:
            messagebox.showerror("Missing input", "Please select an audio file.")
            return
        if not out:
            messagebox.showerror("Missing output", "Please specify an output MIDI path.")
            return
        self._set_busy(True)
        threading.Thread(
            target=self._convert_thread,
            args=(inp, out, self._tempo.get()),
            daemon=True,
        ).start()

    def _convert_thread(self, inp: str, out: str, tempo: int):
        try:
            self._post_progress(10, "Loading audio…")
            y, sr = load_audio(inp)
            duration = len(y) / sr
            self._audio_duration = duration

            self._post_progress(20, f"Loaded {duration:.1f}s of audio — detecting pitches…")
            events = detect(y, sr, on_progress=lambda p, m: self._post_progress(20 + int(p * 0.6), m))

            self._events = events
            self._post_progress(85, f"Building MIDI from {len(events)} note(s)…")
            mid = build_midi(events, tempo_bpm=tempo)
            save_midi(mid, out)
            self._midi_file = mid

            self.after(0, self._draw_roll)
            self.after(0, lambda: messagebox.showinfo("Done", f"Saved {len(events)} notes to:\n{out}"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        finally:
            self.after(0, lambda: self._set_busy(False))

    # ── Recording pipeline ────────────────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            out = self._rec_out.get().strip()
            if not out:
                messagebox.showerror("Missing output", "Please specify an output MIDI path.")
                return
            self._recording = True
            self._rec_btn.config(text="■ Stop Recording")
            self._set_busy(True, keep_rec_btn=True)
            self._recorder.start()
            self._post_progress(5, "Recording… click Stop when finished.")
            # Auto-stop after max duration
            self.after(self._rec_dur.get() * 1000, self._auto_stop_record)
        else:
            self._stop_record()

    def _auto_stop_record(self):
        if self._recording:
            self._stop_record()

    def _stop_record(self):
        if not self._recording:
            return
        self._recording = False
        self._rec_btn.config(text="● Start Recording")
        audio = self._recorder.stop()
        if audio is None or len(audio) == 0:
            messagebox.showerror("Error", "No audio captured.")
            self._set_busy(False)
            return
        out = self._rec_out.get().strip()
        tempo = self._rec_tempo.get()
        threading.Thread(
            target=self._process_recording_thread,
            args=(audio, self._recorder.sample_rate, out, tempo),
            daemon=True,
        ).start()

    def _process_recording_thread(self, audio, sr: int, out: str, tempo: int):
        try:
            self._audio_duration = len(audio) / sr
            self._post_progress(20, f"Recorded {self._audio_duration:.1f}s — detecting pitches…")
            events = detect(audio, sr, on_progress=lambda p, m: self._post_progress(20 + int(p * 0.6), m))
            self._events = events

            self._post_progress(85, f"Building MIDI from {len(events)} note(s)…")
            mid = build_midi(events, tempo_bpm=tempo)
            save_midi(mid, out)
            self._midi_file = mid

            self.after(0, self._draw_roll)
            self.after(0, lambda: messagebox.showinfo("Done", f"Saved {len(events)} notes to:\n{out}"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        finally:
            self.after(0, lambda: self._set_busy(False))

    # ── Piano roll ────────────────────────────────────────────────────────────
    def _draw_roll(self):
        canvas = self._roll
        canvas.delete("all")
        if not self._events:
            canvas.create_text(320, 50, text="No notes detected", fill="#6c7086")
            return

        w = canvas.winfo_width() or 640
        h = canvas.winfo_height() or 100
        duration = max(ev.end_sec for ev in self._events) or 1.0
        note_min = min(ev.midi_note for ev in self._events)
        note_max = max(ev.midi_note for ev in self._events)
        note_range = max(note_max - note_min + 1, 12)

        def x(sec): return sec / duration * w
        def y_top(note): return (note_max - note) / note_range * h
        def y_bot(note): return (note_max - note + 1) / note_range * h

        def color(note):
            if note < 48:
                return NOTE_COLORS["bass"]
            elif note < 72:
                return NOTE_COLORS["mid"]
            return NOTE_COLORS["treble"]

        for ev in self._events:
            x1, x2 = x(ev.start_sec), x(ev.end_sec)
            y1, y2 = y_top(ev.midi_note), y_bot(ev.midi_note)
            canvas.create_rectangle(x1, y1, max(x2, x1 + 2), y2, fill=color(ev.midi_note), outline="")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _post_progress(self, pct: int, msg: str):
        self.after(0, lambda p=pct, m=msg: self._update_progress(p, m))

    def _update_progress(self, pct: int, msg: str):
        self._status.set(msg)
        self._bar["value"] = pct

    def _set_busy(self, busy: bool, keep_rec_btn: bool = False):
        state = "disabled" if busy else "normal"
        for tab in (self._file_tab if hasattr(self, "_file_tab") else [], ):
            pass
        # Walk all interactive children in both tabs
        for widget in self.winfo_children():
            self._set_widget_state(widget, state, keep_rec_btn)
        if not busy:
            self._status.set("Ready.")
            self._bar["value"] = 0

    def _set_widget_state(self, widget, state: str, keep_rec_btn: bool):
        try:
            if keep_rec_btn and widget is self._rec_btn:
                return
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state, keep_rec_btn)
