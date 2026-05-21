import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QGroupBox, QFormLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Biomechanical constants ───────────────────────────────────────────────────
# Phalanx length fractions (proximal, middle, distal); must sum to 1.
PHALANX_FRACS = (0.45, 0.30, 0.25)

# Fraction of total angle contributed by each joint.
# Based on natural max flexion: MCP 80°, PIP 100°, DIP 70° → total 250°.
_MAX_DEG = (80.0, 100.0, 70.0)
_TOT_DEG = sum(_MAX_DEG)
JOINT_FRACS = tuple(a / _TOT_DEG for a in _MAX_DEG)
JOINT_NAMES = ("MCP", "PIP", "DIP")

# Stroke that produces a full natural-flex (250°) at a given d
def natural_stroke(d):
    return d * np.radians(_TOT_DEG)


# ── Pneumatics ────────────────────────────────────────────────────────────────

def compute_pneumatics(bore_mm, rod_mm, p_supply_bar, p_back_bar, stroke_mm):
    """
    Double-acting single-rod cylinder.

    Extension: supply pressure on cap end (full bore), back pressure on rod end.
    Retraction: supply pressure on rod end (annulus), back pressure on cap end.

    1 bar = 0.1 N/mm²

    Returns dict with areas (mm²), forces (N), and air volumes per stroke (cm³).
    """
    A_bore    = np.pi / 4.0 * bore_mm ** 2
    A_rod     = np.pi / 4.0 * rod_mm  ** 2
    A_ann     = max(A_bore - A_rod, 0.0)   # annulus (rod end)

    ps = p_supply_bar * 0.1   # N/mm²
    pb = p_back_bar   * 0.1   # N/mm²

    F_ext = ps * A_bore - pb * A_ann    # net extension force
    F_ret = ps * A_ann  - pb * A_bore   # net retraction force

    # Air consumed per stroke at supply pressure (mm³ → cm³)
    V_ext = A_bore * stroke_mm * 1e-3
    V_ret = A_ann  * stroke_mm * 1e-3

    return {
        "A_bore": A_bore, "A_rod": A_rod, "A_ann": A_ann,
        "F_ext": F_ext,   "F_ret": F_ret,
        "V_ext": V_ext,   "V_ret": V_ret,
    }


# ── Tip-force calculation ─────────────────────────────────────────────────────

def compute_tip_force(F_strip, inner, joint_xy, joint_angles, d):
    """
    Effective grip force at the fingertip via the virtual-work / Jacobian method.

    The piston applies F_strip along the outer strip.  A small strip extension δl
    rotates each joint by δθᵢ = fᵢ · δl / d (coupled motion).  By virtual work:

        F_strip · δl = F_tip · δp_tip

    The Jacobian column for a CW rotation δθ at joint i:
        J_i = (p_tip − p_i)_perp_CW = (Δy, −Δx)

    Effective Jacobian (accounting for coupling):
        J_eff = Σᵢ fᵢ · J_i / d

    Grip direction: perpendicular to the distal phalanx, pointing toward the
    inside of the curl (the "palmar" closing direction).

    Key result:  F_tip = F_strip · d / G  where G is a purely geometric factor
    (sum of weighted projected lever arms).  F_tip is DIRECTLY PROPORTIONAL to d.

    Returns (F_tip_N, mechanical_advantage, G_mm, n_grip_xy).
    """
    if d < 1e-10 or not joint_xy:
        return 0.0, 0.0, 0.0, (0.0, 1.0)

    p_tip       = np.array([inner[0][-1], inner[1][-1]])
    alpha_total = -sum(joint_angles)   # distal phalanx direction (CW ⟹ negative)

    # Grip direction: perpendicular to distal phalanx toward the inside of the curl
    n_grip = np.array([np.sin(alpha_total), -np.cos(alpha_total)])

    # Effective Jacobian
    J_eff = np.zeros(2)
    for i, jxy in enumerate(joint_xy):
        dx, dy = p_tip - np.asarray(jxy)
        J_i     = np.array([dy, -dx])              # CW rotation Jacobian column
        J_eff  += JOINT_FRACS[i] * J_i / d

    G          = float(J_eff @ n_grip)             # geometric factor (mm⁻¹ · mm = dimensionless? no — units: mm/mm = 1/d cancels)
    if abs(G) < 1e-10:
        return 0.0, 0.0, 0.0, tuple(n_grip)

    F_tip = abs(F_strip / G)
    MA    = abs(F_tip / F_strip)                   # = |1/G|; typically << 1 (small d, long finger)

    # G expressed as an effective lever length makes more intuitive sense:
    #   F_tip = F_strip × d / L_eff  →  L_eff = d / MA
    return F_tip, MA, abs(d / MA), tuple(n_grip)


# ── Kinematics ────────────────────────────────────────────────────────────────

def compute_kinematics(L1, d, stroke, completion_pct):
    """Returns (theta_total_rad, delta_l)."""
    delta_l = stroke * (completion_pct / 100.0)
    if delta_l < 1e-12 or d < 1e-12:
        return 0.0, delta_l
    return delta_l / d, delta_l


def build_finger_points(L1, d, theta_total, n_per_seg=300):
    """
    Returns (inner_xy, outer_xy, joint_positions, joint_angles_rad).

    Inner strip — 3 piecewise-straight phalanges with concentrated bends at
    MCP, PIP, DIP joints.  Curls clockwise (downward) from the fixed end.

    Outer strip — parallel curve offset d outward, with a circular arc of
    radius d at each joint corner. The arc length at each corner equals
    d × θ_joint, giving the total extra length Δl = d × θ_total.
    """
    segs   = [f * L1 for f in PHALANX_FRACS]
    thetas = [f * theta_total for f in JOINT_FRACS]

    if theta_total < 1e-12:
        l = [f * L1 for f in PHALANX_FRACS]
        joint_xy_straight = [
            np.array([0.0,              0.0]),
            np.array([l[0],             0.0]),
            np.array([l[0] + l[1],      0.0]),
        ]
        t = np.linspace(0.0, 1.0, n_per_seg * 3)
        return (t * L1, np.zeros_like(t)), (t * L1, np.full_like(t, d)), joint_xy_straight, thetas

    # ── Inner strip ──────────────────────────────────────────────────────────
    inner_pts = [(0.0, 0.0)]
    joint_xy  = []
    pos       = np.array([0.0, 0.0])
    alpha     = 0.0          # current direction (rad); 0 = rightward, CW = decreasing

    for i in range(3):
        joint_xy.append(pos.copy())
        alpha -= thetas[i]   # CW bend at joint i
        dv     = np.array([np.cos(alpha), np.sin(alpha)])
        for t in np.linspace(0.0, 1.0, n_per_seg + 1)[1:]:
            p = pos + t * segs[i] * dv
            inner_pts.append((float(p[0]), float(p[1])))
        pos += segs[i] * dv

    # ── Outer strip (parallel curve with corner arcs) ─────────────────────────
    # The outward normal for a CW-curling strip is to the LEFT of the travel
    # direction: left-normal at angle α is (−sin α, cos α) = direction α + π/2.
    outer_pts = []
    pos       = np.array([0.0, 0.0])
    alpha     = 0.0

    for i in range(3):
        a0     = alpha          # direction before this bend
        alpha -= thetas[i]
        a1     = alpha          # direction after this bend

        # Corner arc: CW from left-normal-before (a0 + π/2) to left-normal-after
        # (a1 + π/2 = a0 + π/2 − θ_i).  Radius d, centred at inner joint pos.
        n_arc = max(3, round(24 * thetas[i] / np.pi) + 2)
        for ang in np.linspace(a0 + np.pi / 2, a1 + np.pi / 2, n_arc):
            outer_pts.append((pos[0] + d * np.cos(ang), pos[1] + d * np.sin(ang)))

        # Straight segment: offset d to the left of the post-bend direction
        dv = np.array([np.cos(a1), np.sin(a1)])
        ln = np.array([-np.sin(a1), np.cos(a1)])       # left normal
        for t in np.linspace(0.0, 1.0, n_per_seg + 1)[1:]:
            p = pos + t * segs[i] * dv
            outer_pts.append((p[0] + d * ln[0], p[1] + d * ln[1]))

        pos += segs[i] * dv

    ix = np.array([p[0] for p in inner_pts])
    iy = np.array([p[1] for p in inner_pts])
    ox = np.array([p[0] for p in outer_pts])
    oy = np.array([p[1] for p in outer_pts])

    return (ix, iy), (ox, oy), joint_xy, thetas


# ── Parameter widget ──────────────────────────────────────────────────────────

class ParamRow(QWidget):
    """Linked slider + spinbox for a single floating-point parameter."""

    valueChanged = Signal(float)

    def __init__(self, label, lo, hi, default, decimals=1, suffix="", parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(110)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(int(lo * self._scale))
        self._slider.setMaximum(int(hi * self._scale))
        self._slider.setValue(int(default * self._scale))

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setMinimum(lo)
        self._spin.setMaximum(hi)
        self._spin.setValue(default)
        self._spin.setSuffix(suffix)
        self._spin.setFixedWidth(110)

        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)

        row.addWidget(lbl)
        row.addWidget(self._slider, 1)
        row.addWidget(self._spin)

    def _from_slider(self, raw):
        val = raw / self._scale
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self.valueChanged.emit(val)

    def _from_spin(self, val):
        self._slider.blockSignals(True)
        self._slider.setValue(round(val * self._scale))
        self._slider.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self):
        return self._spin.value()


# ── Matplotlib canvas ─────────────────────────────────────────────────────────

BG        = "#1e1e2e"
AX_BG     = "#24273a"
INNER_COL = "#89b4fa"   # blue  – ventral strip / finger surface
OUTER_COL = "#f38ba8"   # pink  – outer actuating strip
BODY_COL  = "#45475a"   # fill between strips
JOINT_COL = "#a6e3a1"   # green – joint markers
GRID_COL  = "#363a4f"
TEXT_COL  = "#cdd6f4"


class MechCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor=BG, tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    def redraw(self, inner, outer, joint_xy, joint_angles, theta_total, completion_pct,
               tip_force_N=0.0, n_grip=(0.0, 1.0)):
        ax = self.ax
        ax.clear()
        self._style_axes()

        ix, iy = inner
        ox, oy = outer

        # Fill finger body between the two strips
        fill_x = np.concatenate([ox, ix[::-1]])
        fill_y = np.concatenate([oy, iy[::-1]])
        ax.fill(fill_x, fill_y, color=BODY_COL, alpha=0.55, zorder=1)

        # Strips
        ax.plot(ix, iy, color=INNER_COL, linewidth=2.0,
                label="Inner strip (ventral)", zorder=3)
        ax.plot(ox, oy, color=OUTER_COL, linewidth=2.0,
                label="Outer strip (actuator)", zorder=3)

        # Fixed-end anchor
        ax.plot([ix[0], ox[0]], [iy[0], oy[0]],
                color=JOINT_COL, linewidth=2.5, solid_capstyle="round",
                zorder=4, label="Fixed end")

        # Joint markers and labels
        offset_scale = max(abs(ix).max(), abs(iy).max(), 1.0) * 0.04
        for i, (jx, jy) in enumerate(joint_xy):
            ax.plot(jx, jy, "o", color=JOINT_COL, markersize=9, zorder=6)
            ax.annotate(
                f"{JOINT_NAMES[i]}\n{np.degrees(joint_angles[i]):.1f}°",
                xy=(jx, jy),
                xytext=(jx - offset_scale * 0.5, jy + offset_scale * 2.5),
                color=JOINT_COL, fontsize=7.5,
                ha="center", va="bottom", zorder=7,
            )

        # Free-end markers
        ax.plot(ix[-1], iy[-1], "D", color=INNER_COL, markersize=6, zorder=5)
        ax.plot(ox[-1], oy[-1], "D", color=OUTER_COL, markersize=6, zorder=5)

        # Tip force arrow (grip direction, scaled to ~15% of finger length)
        if tip_force_N > 1e-3:
            span  = max(np.ptp(ix), np.ptp(iy), 1.0)
            scale = 0.18 * span / max(tip_force_N, 1.0)   # arrow length per N
            ax.annotate(
                f"{tip_force_N:.1f} N",
                xy        = (ix[-1] + n_grip[0] * tip_force_N * scale,
                             iy[-1] + n_grip[1] * tip_force_N * scale),
                xytext    = (ix[-1], iy[-1]),
                arrowprops= dict(arrowstyle="-|>", color="#fab387",
                                 lw=2.0, mutation_scale=14),
                color="#fab387", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=8,
            )

        theta_deg = np.degrees(theta_total)
        title = (
            f"θ_total = {theta_deg:.1f}°  "
            f"(MCP {np.degrees(joint_angles[0]):.0f}° / "
            f"PIP {np.degrees(joint_angles[1]):.0f}° / "
            f"DIP {np.degrees(joint_angles[2]):.0f}°)"
            if theta_total > 1e-12
            else "Finger straight  (0 % completion)"
        )
        ax.set_title(title, color=TEXT_COL, fontsize=9, pad=8)

        ax.legend(facecolor="#313244", edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, fontsize=8, loc="best")

        self.fig.canvas.draw_idle()


# ── Output value label ────────────────────────────────────────────────────────

class OutLabel(QLabel):
    def __init__(self):
        super().__init__("—")
        self.setStyleSheet("color: #89dceb; font-weight: bold;")


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Mechanism Kinematics")
        self.resize(1200, 720)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Left panel ──────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        # Parameters
        p_group = QGroupBox("Parameters")
        p_layout = QVBoxLayout(p_group)
        p_layout.setSpacing(6)

        self.p_L1         = ParamRow("Inner length",  10,   200,  85, decimals=0, suffix=" mm")
        self.p_d          = ParamRow("Strip offset d",  0.5,  20,   2, decimals=1, suffix=" mm")
        self.p_stroke     = ParamRow("Stroke",          0,   100,  10, decimals=1, suffix=" mm")
        self.p_completion = ParamRow("Completion",      0,   100,   0, decimals=0, suffix=" %")

        for w in (self.p_L1, self.p_d, self.p_stroke, self.p_completion):
            p_layout.addWidget(w)
            w.valueChanged.connect(self._refresh)

        p_layout.addWidget(self._note(
            "Outer length = inner length + stroke × completion\n"
            "θ_total = Δl / d  →  distributed across MCP / PIP / DIP"
        ))
        left_layout.addWidget(p_group)

        # Pneumatic actuator parameters
        pn_group = QGroupBox("Pneumatic Actuator")
        pn_layout = QVBoxLayout(pn_group)
        pn_layout.setSpacing(6)

        self.p_bore   = ParamRow("Bore dia.",    3,  20, 16, decimals=1, suffix=" mm")
        self.p_rod    = ParamRow("Rod dia.",     1,  20,  6, decimals=1, suffix=" mm")
        self.p_supply = ParamRow("Supply press.",0,  10,  6, decimals=2, suffix=" bar")

        for w in (self.p_bore, self.p_rod, self.p_supply):
            pn_layout.addWidget(w)
            w.valueChanged.connect(self._refresh)

        pn_layout.addWidget(self._note(
            "Double-acting, single-rod cylinder. Exhaust to atmosphere."
        ))
        left_layout.addWidget(pn_group)

        # Outputs
        o_group = QGroupBox("Outputs")
        o_layout = QFormLayout(o_group)
        o_layout.setHorizontalSpacing(12)
        o_layout.setVerticalSpacing(2)
        o_layout.setContentsMargins(6, 6, 6, 6)

        def _sep(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #6c7086; font-size: 9px; margin-top: 3px;")
            return lbl

        # ── Kinematics
        self.o_theta     = OutLabel()
        self.o_mcp       = OutLabel()
        self.o_pip       = OutLabel()
        self.o_dip       = OutLabel()
        self.o_end_inner = OutLabel()

        o_layout.addRow(_sep("— Kinematics —"))
        o_layout.addRow("Total angle (θ):",       self.o_theta)
        o_layout.addRow("  MCP:",                 self.o_mcp)
        o_layout.addRow("  PIP:",                 self.o_pip)
        o_layout.addRow("  DIP:",                 self.o_dip)
        o_layout.addRow("Free end (x, y):",       self.o_end_inner)

        # ── Pneumatics
        self.o_A_bore  = OutLabel()
        self.o_A_ann   = OutLabel()
        self.o_F_ext   = OutLabel()
        self.o_F_ret   = OutLabel()
        self.o_F_ratio = OutLabel()
        self.o_V_ext   = OutLabel()
        self.o_V_ret   = OutLabel()

        o_layout.addRow(_sep("— Pneumatics —"))
        o_layout.addRow("Bore / annulus area:",   self.o_A_bore)
        o_layout.addRow("Extension force:",       self.o_F_ext)
        o_layout.addRow("Retraction force:",      self.o_F_ret)
        o_layout.addRow("Force ratio (ext/ret):", self.o_F_ratio)
        o_layout.addRow("Air vol. ext / ret:",    self.o_V_ext)

        # ── Tip force
        self.o_tip_ext = OutLabel()
        self.o_tip_ret = OutLabel()
        self.o_MA      = OutLabel()
        self.o_L_eff   = OutLabel()

        o_layout.addRow(_sep("— Tip force (virtual work) —"))
        o_layout.addRow("Tip force (flexion):",   self.o_tip_ext)
        o_layout.addRow("Tip force (extension):", self.o_tip_ret)
        o_layout.addRow("Mech. advantage:",       self.o_MA)
        o_layout.addRow("Eff. lever arm:",        self.o_L_eff)

        left_layout.addWidget(o_group)

        left_layout.addWidget(self._note(
            "Joint model — MCP 80° : PIP 100° : DIP 70°  (max 250° total)\n"
            "Phalanges — proximal 45% : middle 30% : distal 25%"
        ))
        left_layout.addStretch()

        # ── Canvas ──────────────────────────────────────────────────────────
        self.canvas = MechCanvas()

        layout.addWidget(left)
        layout.addWidget(self.canvas, 1)

    @staticmethod
    def _note(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #6c7086; font-size: 9px; padding: 1px 0;")
        return lbl

    def _refresh(self):
        L1         = self.p_L1.value()
        d          = self.p_d.value()
        stroke     = self.p_stroke.value()
        completion = self.p_completion.value()

        theta_total, delta_l = compute_kinematics(L1, d, stroke, completion)
        inner, outer, joint_xy, joint_angles = build_finger_points(L1, d, theta_total)

        end_x, end_y = inner[0][-1], inner[1][-1]

        self.o_theta.setText(f"{np.degrees(theta_total):.2f}°")
        self.o_mcp.setText(f"{np.degrees(joint_angles[0]):.2f}°")
        self.o_pip.setText(f"{np.degrees(joint_angles[1]):.2f}°")
        self.o_dip.setText(f"{np.degrees(joint_angles[2]):.2f}°")
        self.o_end_inner.setText(f"({end_x:.1f},  {end_y:.1f}) mm")

        # ── Pneumatics ──────────────────────────────────────────────────────
        pn = compute_pneumatics(
            bore_mm      = self.p_bore.value(),
            rod_mm       = self.p_rod.value(),
            p_supply_bar = self.p_supply.value(),
            p_back_bar   = 0.0,
            stroke_mm    = stroke,
        )

        def _f(n):
            return f"{n:.1f} N  ({n / 9.81:.2f} kgf)"

        self.o_A_bore.setText(
            f"{pn['A_bore']:.1f} / {pn['A_ann']:.1f} mm²")
        self.o_F_ext.setText(_f(pn['F_ext']))
        self.o_F_ret.setText(_f(pn['F_ret']))

        if pn['F_ret'] > 1e-6:
            self.o_F_ratio.setText(f"{pn['F_ext'] / pn['F_ret']:.3f}")
        else:
            self.o_F_ratio.setText("— (zero retraction)")

        self.o_V_ext.setText(
            f"{pn['V_ext']:.2f} / {pn['V_ret']:.2f} cm³")

        # ── Tip force (virtual work) ─────────────────────────────────────────
        tip_ext, MA, L_eff, n_grip = compute_tip_force(
            pn['F_ext'], inner, joint_xy, joint_angles, d)
        tip_ret, _,   _,     _      = compute_tip_force(
            pn['F_ret'], inner, joint_xy, joint_angles, d)

        self.o_tip_ext.setText(_f(tip_ext))
        self.o_tip_ret.setText(_f(tip_ret))
        self.o_MA.setText(f"{MA:.4f}  (= d / L_eff)")
        self.o_L_eff.setText(f"{L_eff:.2f} mm")

        self.canvas.redraw(inner, outer, joint_xy, joint_angles,
                           theta_total, completion, tip_ext, n_grip)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    dark  = QColor("#1e1e2e")
    mid   = QColor("#313244")
    text  = QColor("#cdd6f4")
    palette.setColor(QPalette.Window,          dark)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            mid)
    palette.setColor(QPalette.AlternateBase,   dark)
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          mid)
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.Highlight,       QColor("#89b4fa"))
    palette.setColor(QPalette.HighlightedText, dark)
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
