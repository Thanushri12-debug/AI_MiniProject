"""
app.py  –  Bayesian Disease Predictor  •  Tkinter Frontend
A sleek, dark medical-themed GUI for symptom-based disease prediction.

Run:  python app.py
Requires: model.pkl, label_encoder.pkl, features.pkl in backend/
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, font
import threading
from turtle import left

# Ensure backend is importable regardless of where the script is run from
_HERE    = os.path.dirname(os.path.abspath(__file__))          # .../frontend/
_ROOT    = os.path.dirname(_HERE)                               # .../AIMiniProject/
_BACKEND = os.path.join(_ROOT, "backend")                      # .../backend/
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BACKEND)
from predictor import BayesianDiseasePredictor, DISEASE_INFO

# ── Palette ───────────────────────────────────────────────────────────────────
BG_DARK    = "#080E1A"
BG_PANEL   = "#0F1825"
BG_CARD    = "#141E2E"
BG_HOVER   = "#1A2540"
ACCENT     = "#00D4FF"
ACCENT2    = "#0094B3"
SUCCESS    = "#00E676"
WARNING    = "#FFB74D"
DANGER     = "#FF5252"
TEXT_HI    = "#E8F4FD"
TEXT_MID   = "#8BAFC5"
TEXT_LO    = "#3A5068"
BORDER     = "#1E3048"
GOLD       = "#FFD54F"

# ── Symptom definitions ───────────────────────────────────────────────────────
SYMPTOMS = [
    ("fever",          "Fever",          "🌡️"),
    ("cough",          "Cough",          "😮‍💨"),
    ("headache",       "Headache",       "🤕"),
    ("fatigue",        "Fatigue",        "😴"),
    ("body_aches",     "Body Aches",     "💪"),
    ("chills",         "Chills",         "🥶"),
    ("sore_throat",    "Sore Throat",    "🤧"),
    ("runny_nose",     "Runny Nose",     "👃"),
    ("nausea",         "Nausea",         "🤢"),
    ("sweating",       "Sweating",       "💧"),
    ("joint_pain",     "Joint Pain",     "🦴"),
    ("rash",           "Rash",           "🔴"),
    ("vomiting",       "Vomiting",       "🤮"),
    ("loss_of_appetite","Loss of Appetite","🍽️"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  ROUNDED CANVAS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def round_rect(canvas, x1, y1, x2, y2, r=12, **kw):
    """Draw a rounded rectangle on a Canvas."""
    pts = [
        x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
        x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)

# ─────────────────────────────────────────────────────────────────────────────
#  SYMPTOM TOGGLE BUTTON
# ─────────────────────────────────────────────────────────────────────────────

class SymptomButton(tk.Frame):
    def __init__(self, parent, key, label, icon, var, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.key  = key
        self.var  = var
        self.icon = icon
        self.lbl  = label
        self._active = False

        self.canvas = tk.Canvas(self, width=148, height=54,
                                bg=BG_PANEL, highlightthickness=0)
        self.canvas.pack()
        self._draw(False)

        self.canvas.bind("<Button-1>", self._toggle)
        self.canvas.bind("<Enter>",    self._on_enter)
        self.canvas.bind("<Leave>",    self._on_leave)

    def _draw(self, active):
        self.canvas.delete("all")
        fill   = ACCENT if active else BG_CARD
        outline= ACCENT if active else BORDER
        text_c = BG_DARK if active else TEXT_MID
        icon_c = BG_DARK if active else ACCENT

        round_rect(self.canvas, 3, 3, 145, 51, r=10,
                   fill=fill, outline=outline, width=1 if not active else 0)
        # glow ring when active
        if active:
            round_rect(self.canvas, 1, 1, 147, 53, r=11,
                       fill="", outline=ACCENT2, width=1)

        self.canvas.create_text(22, 27, text=self.icon, font=("Segoe UI Emoji", 14),
                                fill=icon_c, anchor="center")
        self.canvas.create_text(83, 27, text=self.lbl,
                                font=("Courier", 11, "bold" if active else "normal"),
                                fill=text_c, anchor="center")

    def _toggle(self, e=None):
        self._active = not self._active
        self.var.set(1 if self._active else 0)
        self._draw(self._active)

    def _on_enter(self, e=None):
        if not self._active:
            self.canvas.delete("all")
            round_rect(self.canvas, 3, 3, 145, 51, r=10,
                       fill=BG_HOVER, outline=ACCENT2, width=1)
            self.canvas.create_text(22, 27, text=self.icon,
                                    font=("Segoe UI Emoji", 14), fill=ACCENT2)
            self.canvas.create_text(83, 27, text=self.lbl,
                                    font=("Courier", 11), fill=TEXT_HI, anchor="center")

    def _on_leave(self, e=None):
        if not self._active:
            self._draw(False)

    def reset(self):
        self._active = False
        self.var.set(0)
        self._draw(False)

# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATED PROBABILITY BAR
# ─────────────────────────────────────────────────────────────────────────────

class ProbBar(tk.Frame):
    def __init__(self, parent, disease, color, **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self.disease = disease
        self.color   = color
        self._pct    = 0.0
        self._target = 0.0

        info = DISEASE_INFO.get(disease, {})

        row1 = tk.Frame(self, bg=BG_CARD)
        row1.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(row1, text=f"{info.get('icon','🏥')}  {disease}",
                 font=("Courier", 13, "bold"), fg=TEXT_HI, bg=BG_CARD).pack(side="left")
        self.pct_lbl = tk.Label(row1, text="0.00%",
                                font=("Courier", 13, "bold"), fg=color, bg=BG_CARD)
        self.pct_lbl.pack(side="right")

        bar_frame = tk.Frame(self, bg=BORDER, height=10, relief="flat")
        bar_frame.pack(fill="x", padx=16, pady=(0, 12))
        bar_frame.pack_propagate(False)

        self.bar = tk.Frame(bar_frame, bg=color, height=10)
        self.bar.place(x=0, y=0, relheight=1, width=0)
        self._bar_w = 0

        bar_frame.bind("<Configure>", lambda e: setattr(self, "_bar_w", e.width))

    def animate_to(self, target_pct):
        self._target = target_pct
        self._step()

    def _step(self):
        diff = self._target - self._pct
        if abs(diff) < 0.2:
            self._pct = self._target
        else:
            self._pct += diff * 0.15
        w = int(self._bar_w * self._pct / 100)
        self.bar.place(x=0, y=0, relheight=1, width=max(0, w))
        self.pct_lbl.configure(text=f"{self._pct:.2f}%")
        if abs(self._target - self._pct) > 0.1:
            self.after(16, self._step)

    def reset(self):
        self._pct = self._target = 0.0
        self.bar.place(x=0, y=0, relheight=1, width=0)
        self.pct_lbl.configure(text="0.00%")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class DiseasePredictorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("BayesMD — Disease Predictor")
        self.geometry("1100x820")
        self.minsize(960, 720)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Load model
        self.predictor = BayesianDiseasePredictor()

        # Symptom variables
        self.sym_vars = {k: tk.IntVar(value=0) for k, *_ in SYMPTOMS}

        self._build_ui()
        self._center_window()

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=BG_PANEL, height=64)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="⚕", font=("Segoe UI Emoji", 24),
                 fg=ACCENT, bg=BG_PANEL).pack(side="left", padx=20, pady=12)
        tk.Label(bar, text="BayesMD",
                 font=("Courier", 20, "bold"), fg=TEXT_HI, bg=BG_PANEL).pack(side="left")
        tk.Label(bar, text="  Bayesian Disease Prediction System",
                 font=("Courier", 12), fg=TEXT_MID, bg=BG_PANEL).pack(side="left", padx=10)

        tk.Label(bar, text="Naive Bayes  |  scikit-learn  |  Python",
                 font=("Courier", 10), fg=TEXT_LO, bg=BG_PANEL).pack(side="right", padx=20)

        # ── Separator ────────────────────────────────────────────────────────
        sep = tk.Frame(self, bg=ACCENT, height=2)
        sep.pack(fill="x")

        # ── Body: left + right ────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Title
        tk.Label(
            left,
            text="S E L E C T   S Y M P T O M S",
            font=("Courier", 11, "bold"),   
            fg=ACCENT,
            bg=BG_DARK
        ).pack(anchor="w", pady=(0, 10))
        # Subtitle
        tk.Label(left, text="Click the symptoms you are experiencing:",
                 font=("Courier", 11), fg=TEXT_MID, bg=BG_DARK).pack(anchor="w", pady=(0, 12))

        # Symptom grid
        grid = tk.Frame(left, bg=BG_DARK)
        grid.pack(fill="x")
        self.sym_buttons = {}
        COLS = 4
        for i, (key, label, icon) in enumerate(SYMPTOMS):
            r, c = divmod(i, COLS)
            btn = SymptomButton(grid, key, label, icon, self.sym_vars[key])
            btn.grid(row=r, column=c, padx=4, pady=4, sticky="w")
            self.sym_buttons[key] = btn

        # Buttons row
        btn_row = tk.Frame(left, bg=BG_DARK)
        btn_row.pack(fill="x", pady=16)

        self.predict_btn = tk.Button(
            btn_row, text="  PREDICT DISEASE  ",
            font=("Courier", 13, "bold"),
            fg=BG_DARK, bg=ACCENT,
            activebackground=ACCENT2, activeforeground=BG_DARK,
            relief="flat", cursor="hand2", padx=10, pady=10,
            command=self._predict
        )
        self.predict_btn.pack(side="left", padx=(0, 12))

        tk.Button(
            btn_row, text="  CLEAR ALL  ",
            font=("Courier", 12),
            fg=TEXT_MID, bg=BG_CARD,
            activebackground=BG_HOVER, activeforeground=TEXT_HI,
            relief="flat", cursor="hand2", padx=10, pady=10,
            command=self._clear
        ).pack(side="left")

        # Active symptoms display
        tk.Label(left, text="ACTIVE SYMPTOMS:",
                 font=("Courier", 10, "bold"), fg=TEXT_LO, bg=BG_DARK).pack(anchor="w")
        self.active_lbl = tk.Label(left, text="None selected",
                                   font=("Courier", 11), fg=ACCENT2, bg=BG_DARK,
                                   wraplength=560, justify="left")
        self.active_lbl.pack(anchor="w", pady=(4, 0))

        # Bayesian formula display
        formula_frame = tk.Frame(left, bg=BG_CARD, relief="flat")
        formula_frame.pack(fill="x", pady=(14, 0))

        tk.Label(formula_frame, text="BAYES' THEOREM APPLIED:",
                 font=("Courier", 10, "bold"), fg=ACCENT, bg=BG_CARD).pack(anchor="w", padx=14, pady=(10,4))

        self.formula_lbl = tk.Label(
            formula_frame,
            text="P(Disease | Symptoms)  =  P(Symptoms | Disease) × P(Disease)\n"
                 "                                    ─────────────────────────────────\n"
                 "                                               P(Symptoms)",
            font=("Courier", 10), fg=TEXT_MID, bg=BG_CARD,
            justify="left"
        )
        self.formula_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        self.bayes_detail = tk.Label(
            formula_frame, text="← Run a prediction to see live Bayesian calculations",
            font=("Courier", 10), fg=TEXT_LO, bg=BG_CARD, justify="left", wraplength=520
        )
        self.bayes_detail.pack(anchor="w", padx=14, pady=(0, 12))

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="PREDICTION RESULTS",
                 font=("Courier", 11, "bold"), fg=ACCENT, bg=BG_DARK).pack(anchor="w", pady=(0, 10))

        # Top disease card
        self.top_card = tk.Frame(right, bg=BG_CARD, relief="flat")
        self.top_card.pack(fill="x", pady=(0, 12))

        self.top_icon  = tk.Label(self.top_card, text="🏥", font=("Segoe UI Emoji", 36),
                                   fg=TEXT_HI, bg=BG_CARD)
        self.top_icon.pack(pady=(18, 4))
        self.top_name  = tk.Label(self.top_card, text="Awaiting Input",
                                   font=("Courier", 18, "bold"), fg=TEXT_MID, bg=BG_CARD)
        self.top_name.pack()
        self.top_conf  = tk.Label(self.top_card, text="—",
                                   font=("Courier", 13), fg=TEXT_LO, bg=BG_CARD)
        self.top_conf.pack(pady=(2, 4))
        self.risk_lbl  = tk.Label(self.top_card, text="",
                                   font=("Courier", 11, "bold"), bg=BG_CARD)
        self.risk_lbl.pack(pady=(0, 4))
        self.desc_lbl  = tk.Label(self.top_card, text="Select symptoms and click PREDICT",
                                   font=("Courier", 10), fg=TEXT_MID, bg=BG_CARD,
                                   wraplength=320, justify="center")
        self.desc_lbl.pack(pady=(0, 6))
        self.advice_lbl = tk.Label(self.top_card, text="",
                                    font=("Courier", 10), fg=WARNING, bg=BG_CARD,
                                    wraplength=320, justify="center")
        self.advice_lbl.pack(pady=(0, 16))

        # Probability bars
        tk.Label(right, text="PROBABILITY BREAKDOWN",
                 font=("Courier", 10, "bold"), fg=TEXT_LO, bg=BG_DARK).pack(anchor="w", pady=(0, 8))

        self.prob_bars = {}
        colors = {"Flu": "#4FC3F7", "Malaria": "#FF7043", "Dengue": "#EF5350", "Cold": "#66BB6A"}
        for disease, color in colors.items():
            bar = ProbBar(right, disease, color)
            bar.pack(fill="x", pady=3)
            self.prob_bars[disease] = bar

        # Status bar
        self.status = tk.Label(right, text="Ready",
                                font=("Courier", 10), fg=TEXT_LO, bg=BG_DARK)
        self.status.pack(anchor="w", pady=(10, 0))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _predict(self):
        selected = [k for k, v in self.sym_vars.items() if v.get() == 1]

        if not selected:
            self.status.configure(text="⚠  Please select at least one symptom.", fg=WARNING)
            return

        # Update active label
        display = "  •  ".join(s.replace("_", " ").title() for s in selected)
        self.active_lbl.configure(text=display)

        self.status.configure(text="Calculating posterior probabilities...", fg=ACCENT)
        self.predict_btn.configure(state="disabled", text="  PREDICTING...  ")

        def run():
            result = self.predictor.predict(selected)
            self.after(0, lambda: self._show_result(result))

        threading.Thread(target=run, daemon=True).start()

    def _show_result(self, result):
        top   = result["top_disease"]
        probs = result["probabilities"]
        info  = DISEASE_INFO.get(top, {})
        risk  = result["risk"]

        # Top card
        self.top_icon.configure(text=result["icon"])
        self.top_name.configure(text=top, fg=result["color"])
        self.top_conf.configure(text=f"Confidence: {probs[top]:.2f}%", fg=result["color"])

        risk_colors = {"Low": SUCCESS, "Moderate": WARNING, "High": DANGER}
        self.risk_lbl.configure(text=f"Risk Level: {risk}",
                                fg=risk_colors.get(risk, TEXT_MID))
        self.desc_lbl.configure(text=result["description"])
        self.advice_lbl.configure(text=f"ℹ  {result['advice']}")

        # Animate bars
        for disease, bar in self.prob_bars.items():
            bar.animate_to(probs.get(disease, 0))

        # Bayesian detail
        fs = result["formula_steps"]
        prior_str = "  ".join(f"{d}: {v}%" for d, v in fs["prior_probabilities"].items())
        detail = (
            f"Prior P(Disease):  {prior_str}\n"
            f"Likelihoods:       {', '.join(f'{d}: {v:.4f}%' for d,v in fs['likelihoods'].items())}\n"
            f"→  Posteriors:     {', '.join(f'{d}: {v:.2f}%' for d,v in fs['posterior_probabilities'].items())}"
        )
        self.bayes_detail.configure(text=detail, fg=TEXT_MID)

        self.status.configure(
            text=f"✓  Prediction complete  —  {top} ({probs[top]:.2f}%)",
            fg=SUCCESS
        )
        self.predict_btn.configure(state="normal", text="  PREDICT DISEASE  ")

    def _clear(self):
        for btn in self.sym_buttons.values():
            btn.reset()
        for bar in self.prob_bars.values():
            bar.reset()
        self.active_lbl.configure(text="None selected")
        self.top_icon.configure(text="🏥")
        self.top_name.configure(text="Awaiting Input", fg=TEXT_MID)
        self.top_conf.configure(text="—", fg=TEXT_LO)
        self.risk_lbl.configure(text="")
        self.desc_lbl.configure(text="Select symptoms and click PREDICT")
        self.advice_lbl.configure(text="")
        self.bayes_detail.configure(
            text="← Run a prediction to see live Bayesian calculations", fg=TEXT_LO
        )
        self.status.configure(text="Cleared", fg=TEXT_LO)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DiseasePredictorApp()
    app.mainloop()
