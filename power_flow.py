"""
power_flow.py — Simulador de Flujo de Potencia en Redes Eléctricas
===================================================================
Implementación completa del método Newton-Raphson para análisis de
flujo de potencia (Load Flow) en sistemas eléctricos de potencia.

CONTENIDO:
  1. YbusBuilder   — Construcción de la matriz de admitancia nodal
  2. PowerFlowSolver — Algoritmo Newton-Raphson completo
  3. Visualización  — Gráficas de convergencia y perfil de voltajes
  4. Sistemas de prueba — IEEE 5-Bus, IEEE 14-Bus
  5. main()        — Punto de entrada con CLI y carga de CSV

USO:
  python power_flow.py                        # IEEE 5-Bus (default)
  python power_flow.py --system ieee14        # IEEE 14-Bus
  python power_flow.py --csv buses.csv lines.csv
  python power_flow.py --load 1.20 --tol 1e-8

FORMATO CSV:
  buses.csv : bus_id, type, V_mag, V_ang, P_gen, Q_gen, P_load, Q_load
  lines.csv : from_bus, to_bus, R, X, B_total
  (valores en p.u.; V_ang en grados; tipo: slack | PV | PQ)

BASE TEÓRICA:
  Glover, Sarma & Overbye — "Power System Analysis and Design", 5th ed.
  Bergen & Vittal — "Power Systems Analysis", 2nd ed.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — MATRIZ DE ADMITANCIA NODAL (Ybus)
# ══════════════════════════════════════════════════════════════════════════════

class YbusBuilder:
    """
    Construye la matriz de admitancia nodal Ybus usando el modelo π de líneas.

    La Ybus es una matriz compleja N×N donde cada elemento representa la
    admitancia eléctrica entre nodos. Se construye aplicando las siguientes
    reglas para cada línea (i → j) con admitancia serie y_s = 1/(R+jX)
    y susceptancia shunt y_sh = jB/2:

        Y_ii += y_s + y_sh    (diagonal: suma de admitancias propias)
        Y_jj += y_s + y_sh
        Y_ij -= y_s           (fuera de diagonal: negativo de la mutual)
        Y_ji -= y_s           (red pasiva → matriz simétrica)

    Modelo π (representación de líneas de transmisión de longitud media):
        i ──[y_s]── j
        │               │
       [y_sh]         [y_sh]
        │               │
       GND             GND
    """

    @staticmethod
    def build(bus_data: pd.DataFrame, line_data: pd.DataFrame) -> np.ndarray:
        """
        Construye y retorna la matriz Ybus compleja (N×N, complex128).

        Parameters
        ----------
        bus_data : pd.DataFrame
            Columnas requeridas: ['bus_id', 'type', 'V_mag', 'V_ang',
            'P_gen', 'Q_gen', 'P_load', 'Q_load']
        line_data : pd.DataFrame
            Columnas requeridas: ['from_bus', 'to_bus', 'R', 'X', 'B_total']
            Todos los valores en por unidad (pu).

        Returns
        -------
        np.ndarray, shape (N, N), dtype=complex128
        """
        n = len(bus_data)
        bus_index = {bid: i for i, bid in enumerate(bus_data['bus_id'].values)}

        # Validaciones
        for _, line in line_data.iterrows():
            for node in ['from_bus', 'to_bus']:
                if line[node] not in bus_index:
                    raise ValueError(f"Nodo {line[node]} no existe en bus_data")
            if abs(complex(float(line['R']), float(line['X']))) < 1e-12:
                raise ZeroDivisionError(
                    f"Línea {line['from_bus']}→{line['to_bus']}: impedancia nula"
                )

        Ybus = np.zeros((n, n), dtype=np.complex128)

        for _, line in line_data.iterrows():
            i = bus_index[line['from_bus']]
            j = bus_index[line['to_bus']]
            y_serie = 1.0 / complex(float(line['R']), float(line['X']))
            y_shunt = complex(0.0, float(line['B_total']) / 2.0)

            Ybus[i, i] += y_serie + y_shunt
            Ybus[j, j] += y_serie + y_shunt
            Ybus[i, j] -= y_serie
            Ybus[j, i] -= y_serie

        return Ybus

    @staticmethod
    def summary(Ybus: np.ndarray, bus_ids: list) -> pd.DataFrame:
        """Retorna un DataFrame con los elementos no-nulos de Ybus."""
        rows = []
        for i, bi in enumerate(bus_ids):
            for j, bj in enumerate(bus_ids):
                v = Ybus[i, j]
                if abs(v) > 1e-12:
                    rows.append({
                        'Nodo_i': bi, 'Nodo_j': bj,
                        'G (pu)': round(v.real, 6),
                        'B (pu)': round(v.imag, 6),
                        '|Y| (pu)': round(abs(v), 6),
                        'Ang (°)': round(np.degrees(np.angle(v)), 4)
                    })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — SOLVER NEWTON-RAPHSON
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerFlowResult:
    """Contenedor de todos los resultados del flujo de potencia."""
    converged: bool
    iterations: int
    mismatch_history: list       # norma infinita del mismatch por iteración
    bus_results: Optional[pd.DataFrame] = None
    line_results: Optional[pd.DataFrame] = None
    total_losses_MW: float = 0.0
    total_losses_MVAr: float = 0.0
    message: str = ""


class PowerFlowSolver:
    """
    Resuelve el flujo de potencia por el método iterativo de Newton-Raphson.

    FORMULACIÓN MATEMÁTICA:
    ─────────────────────────────────────────────────────────────────────────
    Variables de estado:   x = [θ_PQ, θ_PV, |V|_PQ]
    Ecuaciones de balance: f(x) = [ΔP_no-slack, ΔQ_PQ] = 0

    Ecuaciones de potencia nodal:
      P_i = |V_i|·Σ_j |V_j|·[G_ij·cos(θ_ij) + B_ij·sin(θ_ij)]
      Q_i = |V_i|·Σ_j |V_j|·[G_ij·sin(θ_ij) - B_ij·cos(θ_ij)]

    Iteración Newton-Raphson:
      J(x_k)·Δx = -f(x_k)    →    x_{k+1} = x_k + Δx

    Jacobiana particionada (J1..J4 son derivadas parciales):
      J = | ∂P/∂θ    ∂P/∂|V| |  =  | J1  J2 |
          | ∂Q/∂θ    ∂Q/∂|V| |     | J3  J4 |

    Elementos diagonales (i=j):
      J1_ii = -Q_i - B_ii·|V_i|²
      J2_ii =  P_i/|V_i| + G_ii·|V_i|
      J3_ii =  P_i - G_ii·|V_i|²
      J4_ii =  Q_i/|V_i| - B_ii·|V_i|

    Elementos fuera de diagonal (i≠j), con θ_ij = θ_i - θ_j:
      J1_ij = |V_i||V_j|(G_ij·sin θ_ij - B_ij·cos θ_ij)
      J2_ij = |V_i|(G_ij·cos θ_ij + B_ij·sin θ_ij)
      J3_ij = -|V_i||V_j|(G_ij·cos θ_ij + B_ij·sin θ_ij)
      J4_ij = |V_i|(G_ij·sin θ_ij - B_ij·cos θ_ij)

    Parameters
    ----------
    tol : float
        Tolerancia de convergencia en pu (norma infinita del mismatch).
    max_iter : int
        Número máximo de iteraciones permitidas.
    S_base_MVA : float
        Potencia base del sistema en MVA.
    verbose : bool
        Si True, imprime el progreso iteración a iteración.
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 50,
                 S_base_MVA: float = 100.0, verbose: bool = True):
        self.tol = tol
        self.max_iter = max_iter
        self.S_base = S_base_MVA
        self.verbose = verbose

    def solve(self, bus_data: pd.DataFrame, line_data: pd.DataFrame,
              Ybus: np.ndarray) -> PowerFlowResult:
        """
        Ejecuta el algoritmo Newton-Raphson completo.

        Parameters
        ----------
        bus_data : pd.DataFrame
        line_data : pd.DataFrame
        Ybus : np.ndarray, shape (N, N), complex128

        Returns
        -------
        PowerFlowResult
        """
        n = len(bus_data)
        bus_ids = bus_data['bus_id'].values
        bus_index = {bid: i for i, bid in enumerate(bus_ids)}
        bus_type = bus_data['type'].values

        G = Ybus.real
        B = Ybus.imag

        # Potencia neta programada (generación - carga), en pu
        P_sch = (bus_data['P_gen'].values - bus_data['P_load'].values).astype(float)
        Q_sch = (bus_data['Q_gen'].values - bus_data['Q_load'].values).astype(float)

        # Flat start: |V|=1.0, θ=0° para nodos PQ; PV mantiene |V| especificado
        V_mag = bus_data['V_mag'].values.copy().astype(float)
        V_ang = np.radians(bus_data['V_ang'].values.copy().astype(float))
        for i, t in enumerate(bus_type):
            if t == 'PQ':
                V_mag[i] = 1.0
                V_ang[i] = 0.0

        idx_non_slack = [i for i, t in enumerate(bus_type) if t != 'slack']
        idx_PQ        = [i for i, t in enumerate(bus_type) if t == 'PQ']

        if self.verbose:
            print(f"\n{'═'*62}")
            print(f"  FLUJO DE POTENCIA — NEWTON-RAPHSON")
            n_slack = sum(1 for t in bus_type if t == 'slack')
            n_pv    = sum(1 for t in bus_type if t == 'PV')
            n_pq    = sum(1 for t in bus_type if t == 'PQ')
            print(f"  {n} nodos | Slack:{n_slack}  PV:{n_pv}  PQ:{n_pq}")
            print(f"  Tolerancia: {self.tol:.2e} pu | Máx: {self.max_iter} iter")
            print(f"{'═'*62}")
            print(f"  {'Iter':>4}  {'||ΔP||∞':>12}  {'||ΔQ||∞':>12}  {'Estado'}")
            print(f"  {'─'*52}")

        mismatch_history = []
        converged = False

        for iteration in range(self.max_iter):
            P_calc, Q_calc = self._power_injections(V_mag, V_ang, G, B, n)

            dP = P_sch[idx_non_slack] - P_calc[idx_non_slack]
            dQ = Q_sch[idx_PQ]        - Q_calc[idx_PQ]

            mm_P = np.max(np.abs(dP)) if len(dP) > 0 else 0.0
            mm_Q = np.max(np.abs(dQ)) if len(dQ) > 0 else 0.0
            mm   = max(mm_P, mm_Q)
            mismatch_history.append(mm)

            if self.verbose:
                flag = "✓ Convergió" if mm < self.tol else "···"
                print(f"  {iteration+1:>4}  {mm_P:>12.6e}  {mm_Q:>12.6e}  {flag}")

            if mm < self.tol:
                converged = True
                break

            J = self._jacobian(V_mag, V_ang, P_calc, Q_calc, G, B,
                               idx_non_slack, idx_PQ, n)
            try:
                delta_x = np.linalg.solve(J, np.concatenate([dP, dQ]))
            except np.linalg.LinAlgError:
                return PowerFlowResult(
                    converged=False, iterations=iteration+1,
                    mismatch_history=mismatch_history,
                    message="Jacobiana singular — verifique los datos de entrada."
                )

            n_ang = len(idx_non_slack)
            for k, i in enumerate(idx_non_slack):
                V_ang[i] += delta_x[k]
            for k, i in enumerate(idx_PQ):
                V_mag[i] += delta_x[n_ang + k]

        if self.verbose:
            print(f"  {'─'*52}")
            msg = f"✓ CONVERGENCIA en {len(mismatch_history)} iteraciones" if converged \
                  else f"✗ NO CONVERGIÓ en {self.max_iter} iteraciones"
            print(f"  {msg}")
            print(f"{'═'*62}\n")

        P_final, Q_final = self._power_injections(V_mag, V_ang, G, B, n)

        bus_results  = self._bus_results(bus_data, V_mag, V_ang, P_final, Q_final, bus_type)
        line_results, Ploss, Qloss = self._line_flows(line_data, V_mag, V_ang, bus_index)

        return PowerFlowResult(
            converged=converged,
            iterations=len(mismatch_history),
            mismatch_history=mismatch_history,
            bus_results=bus_results,
            line_results=line_results,
            total_losses_MW=Ploss * self.S_base,
            total_losses_MVAr=Qloss * self.S_base,
            message="Convergencia exitosa." if converged else "No convergió."
        )

    # ── Métodos internos ───────────────────────────────────────────────────

    def _power_injections(self, Vm, Va, G, B, n):
        """
        Calcula P_i y Q_i con las ecuaciones nodales de potencia:
          P_i = Σ_j |Vi||Vj|(G_ij·cos θ_ij + B_ij·sin θ_ij)
          Q_i = Σ_j |Vi||Vj|(G_ij·sin θ_ij - B_ij·cos θ_ij)
        """
        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for j in range(n):
                th = Va[i] - Va[j]
                c, s = np.cos(th), np.sin(th)
                vv = Vm[i] * Vm[j]
                P[i] += vv * (G[i, j] * c + B[i, j] * s)
                Q[i] += vv * (G[i, j] * s - B[i, j] * c)
        return P, Q

    def _jacobian(self, Vm, Va, Pc, Qc, G, B, idx_ns, idx_pq, n):
        """
        Construye la Jacobiana J = [[J1, J2], [J3, J4]] con las
        derivadas analíticas de P y Q respecto a θ y |V|.
        """
        nns, npq = len(idx_ns), len(idx_pq)
        J1 = np.zeros((nns, nns))
        J2 = np.zeros((nns, npq))
        J3 = np.zeros((npq, nns))
        J4 = np.zeros((npq, npq))

        for r, i in enumerate(idx_ns):
            for c, j in enumerate(idx_ns):
                th = Va[i] - Va[j]
                J1[r, c] = (-Qc[i] - B[i,i]*Vm[i]**2) if i == j else \
                            Vm[i]*Vm[j]*(G[i,j]*np.sin(th) - B[i,j]*np.cos(th))
            for c, j in enumerate(idx_pq):
                th = Va[i] - Va[j]
                J2[r, c] = (Pc[i]/Vm[i] + G[i,i]*Vm[i]) if i == j else \
                            Vm[i]*(G[i,j]*np.cos(th) + B[i,j]*np.sin(th))

        for r, i in enumerate(idx_pq):
            for c, j in enumerate(idx_ns):
                th = Va[i] - Va[j]
                J3[r, c] = (Pc[i] - G[i,i]*Vm[i]**2) if i == j else \
                            -Vm[i]*Vm[j]*(G[i,j]*np.cos(th) + B[i,j]*np.sin(th))
            for c, j in enumerate(idx_pq):
                th = Va[i] - Va[j]
                J4[r, c] = (Qc[i]/Vm[i] - B[i,i]*Vm[i]) if i == j else \
                            Vm[i]*(G[i,j]*np.sin(th) - B[i,j]*np.cos(th))

        return np.block([[J1, J2], [J3, J4]])

    def _bus_results(self, bus_data, Vm, Va, Pc, Qc, bus_type):
        """Organiza los resultados de nodos en un DataFrame."""
        return pd.DataFrame([{
            'Bus':         bus_data['bus_id'].iloc[i],
            'Tipo':        bus_type[i],
            '|V| (pu)':    round(Vm[i], 6),
            'θ (°)':       round(np.degrees(Va[i]), 4),
            'P_iny (MW)':  round(Pc[i] * self.S_base, 4),
            'Q_iny (MVAr)':round(Qc[i] * self.S_base, 4),
            'P_gen (MW)':  round(bus_data['P_gen'].iloc[i] * self.S_base, 3),
            'P_load (MW)': round(bus_data['P_load'].iloc[i] * self.S_base, 3),
        } for i in range(len(bus_data))])

    def _line_flows(self, line_data, Vm, Va, bus_index):
        """
        Calcula flujos y pérdidas en cada línea usando el modelo π:
          I_ij = (Vi - Vj)·y_serie + Vi·(jB/2)
          S_ij = Vi · I_ij*    (potencia enviada)
          S_ji = Vj · I_ji*    (potencia recibida)
          Pérdida = S_ij + S_ji
        """
        rows, Ploss_total, Qloss_total = [], 0.0, 0.0
        for _, line in line_data.iterrows():
            i = bus_index[line['from_bus']]
            j = bus_index[line['to_bus']]
            y_s  = 1.0 / complex(float(line['R']), float(line['X']))
            y_sh = complex(0.0, float(line['B_total']) / 2.0)
            Vi = Vm[i] * np.exp(1j * Va[i])
            Vj = Vm[j] * np.exp(1j * Va[j])
            I_ij = (Vi - Vj) * y_s + Vi * y_sh
            I_ji = (Vj - Vi) * y_s + Vj * y_sh
            S_ij  = Vi * np.conj(I_ij)
            S_ji  = Vj * np.conj(I_ji)
            S_loss = S_ij + S_ji
            Ploss_total += S_loss.real
            Qloss_total += S_loss.imag
            rows.append({
                'De':              line['from_bus'],
                'A':               line['to_bus'],
                'P_envia (MW)':    round(S_ij.real  * self.S_base, 4),
                'Q_envia (MVAr)':  round(S_ij.imag  * self.S_base, 4),
                'P_recibe (MW)':   round(-S_ji.real * self.S_base, 4),
                'Q_recibe (MVAr)': round(-S_ji.imag * self.S_base, 4),
                'P_pérdida (MW)':  round(S_loss.real* self.S_base, 4),
                'Q_pérdida (MVAr)':round(S_loss.imag* self.S_base, 4),
            })
        return pd.DataFrame(rows), Ploss_total, Qloss_total


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(result: PowerFlowResult, system_name: str = "Sistema",
                 save_path: str = None):
    """
    Genera el panel de 4 gráficas del flujo de potencia:
      1. Convergencia Newton-Raphson (escala log)
      2. Perfil de voltajes por nodo
      3. Ángulos de voltaje por nodo
      4. Flujos activos y pérdidas por línea
    """
    DARK = '#0D1117';  PANEL = '#161B22';  TEXT = '#E6EDF3'
    SUB  = '#8B949E';  ACC  = '#F39C12'
    CMAP = {'slack': '#E74C3C', 'PV': '#2ECC71', 'PQ': '#3498DB'}

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK)

    estado  = "Convergió" if result.converged else "NO convergió"
    perdida = f"{result.total_losses_MW:.3f} MW"
    fig.suptitle(
        f'FLUJO DE POTENCIA — {system_name}\n'
        f'Newton-Raphson | {estado} en {result.iterations} iteraciones | '
        f'Pérdidas totales: {perdida}',
        fontsize=13, fontweight='bold', color=TEXT, y=0.98,
        fontfamily='monospace'
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36,
                  left=0.07, right=0.97, top=0.90, bottom=0.10)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(colors=SUB, labelsize=9)
        ax.grid(True, alpha=0.18, color=SUB, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor(SUB); sp.set_alpha(0.3)

    buses  = result.bus_results['Bus'].values
    vmags  = result.bus_results['|V| (pu)'].values
    angs   = result.bus_results['θ (°)'].values
    tipos  = result.bus_results['Tipo'].values
    bcolors = [CMAP.get(t, CMAP['PQ']) for t in tipos]

    # ── 1. Convergencia ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Convergencia Newton-Raphson')
    iters = list(range(1, len(result.mismatch_history) + 1))
    ax1.semilogy(iters, result.mismatch_history, color=ACC, linewidth=2.5,
                 marker='o', markersize=7, markerfacecolor=DARK,
                 markeredgecolor=ACC, markeredgewidth=2)
    ax1.axhline(y=1e-6, color='#E74C3C', lw=1.5, ls='--', alpha=0.8,
                label='Tol. 1e-6')
    ax1.set_xlabel('Iteración', color=SUB, fontsize=10)
    ax1.set_ylabel('||ΔP, ΔQ||∞ (pu)', color=SUB, fontsize=10)
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=SUB, labelcolor=TEXT)

    # ── 2. Perfil de voltajes ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    style_ax(ax2, 'Perfil de Voltajes por Nodo')
    xpos = [str(b) for b in buses]
    bars = ax2.bar(xpos, vmags, color=bcolors, edgecolor=DARK,
                   linewidth=1.5, width=0.65, zorder=3)
    ax2.axhline(y=1.05, color='#E74C3C', lw=1.5, ls='--', alpha=0.75, label='V_max = 1.05')
    ax2.axhline(y=0.95, color='#9B59B6', lw=1.5, ls='--', alpha=0.75, label='V_min = 0.95')
    ax2.axhline(y=1.00, color=SUB, lw=1.0, ls=':', alpha=0.4)
    for bar, vm in zip(bars, vmags):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                 f'{vm:.4f}', ha='center', va='bottom',
                 color=TEXT, fontsize=8.5, fontweight='bold',
                 fontfamily='monospace')
    ax2.set_ylim(0.90, 1.12)
    ax2.set_xlabel('Nodo', color=SUB, fontsize=10)
    ax2.set_ylabel('Magnitud |V| (pu)', color=SUB, fontsize=10)
    legend_elems = [
        mpatches.Patch(facecolor=CMAP['slack'], label='Slack'),
        mpatches.Patch(facecolor=CMAP['PV'],    label='PV (gen)'),
        mpatches.Patch(facecolor=CMAP['PQ'],    label='PQ (carga)'),
        plt.Line2D([0],[0], color='#E74C3C', ls='--', label='V_max=1.05'),
        plt.Line2D([0],[0], color='#9B59B6', ls='--', label='V_min=0.95'),
    ]
    ax2.legend(handles=legend_elems, fontsize=8, facecolor=PANEL,
               edgecolor=SUB, labelcolor=TEXT, ncol=3, loc='lower right')

    # ── 3. Ángulos de voltaje ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Ángulos de Voltaje')
    bars3 = ax3.bar(xpos, angs, color=bcolors, edgecolor=DARK,
                    linewidth=1.5, width=0.65, zorder=3)
    ax3.axhline(y=0, color=SUB, lw=1, ls='-', alpha=0.4)
    for bar, ang in zip(bars3, angs):
        offset = 0.05 if ang >= 0 else -0.25
        ax3.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + offset,
                 f'{ang:.2f}°', ha='center', va='bottom',
                 color=TEXT, fontsize=8, fontfamily='monospace')
    ax3.set_xlabel('Nodo', color=SUB, fontsize=10)
    ax3.set_ylabel('Ángulo θ (°)', color=SUB, fontsize=10)

    # ── 4. Flujos en líneas ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    style_ax(ax4, 'Flujos de Potencia en Líneas')
    if result.line_results is not None and len(result.line_results) > 0:
        lr = result.line_results
        labels = [f"{int(r['De'])}→{int(r['A'])}" for _, r in lr.iterrows()]
        P_send = lr['P_envia (MW)'].values
        P_loss = lr['P_pérdida (MW)'].values.clip(min=0)
        x = np.arange(len(labels)); w = 0.38
        b_send = ax4.bar(x - w/2, P_send, w, color='#3498DB',
                         edgecolor=DARK, linewidth=1.2, zorder=3, label='P enviada (MW)')
        b_loss = ax4.bar(x + w/2, P_loss, w, color='#E74C3C',
                         edgecolor=DARK, linewidth=1.2, zorder=3,
                         alpha=0.85, label='Pérdidas (MW)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=9, fontfamily='monospace')
        ax4.set_xlabel('Línea (De→A)', color=SUB, fontsize=10)
        ax4.set_ylabel('Potencia (MW)', color=SUB, fontsize=10)
        ax4.legend(fontsize=9, facecolor=PANEL, edgecolor=SUB, labelcolor=TEXT)
        for bar, loss, send in zip(b_loss, P_loss, P_send):
            if send > 0.01:
                pct = loss / send * 100
                ax4.text(bar.get_x() + bar.get_width()/2.,
                         bar.get_height() + max(P_send)*0.01,
                         f'{pct:.1f}%', ha='center', va='bottom',
                         color=TEXT, fontsize=7.5, fontfamily='monospace')

    path = save_path or 'power_flow_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f"  📊 Gráfica guardada: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SISTEMAS DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

def ieee5_data():
    """Sistema IEEE de 5 nodos — caso de prueba clásico."""
    buses = pd.DataFrame({
        'bus_id': [1,       2,      3,      4,      5     ],
        'type':   ['slack', 'PV',   'PQ',   'PQ',   'PQ'  ],
        'V_mag':  [1.06,    1.045,  1.0,    1.0,    1.0   ],
        'V_ang':  [0.0,     0.0,    0.0,    0.0,    0.0   ],
        'P_gen':  [0.0,     0.40,   0.0,    0.0,    0.0   ],
        'Q_gen':  [0.0,     0.0,    0.0,    0.0,    0.0   ],
        'P_load': [0.0,     0.20,   0.25,   0.40,   0.50  ],
        'Q_load': [0.0,     0.10,   0.105,  0.05,   0.10  ],
    })
    lines = pd.DataFrame({
        'from_bus': [1,    1,    2,    2,    3   ],
        'to_bus':   [2,    3,    3,    5,    4   ],
        'R':        [0.02, 0.08, 0.06, 0.04, 0.01],
        'X':        [0.06, 0.24, 0.18, 0.12, 0.03],
        'B_total':  [0.06, 0.025,0.02, 0.015,0.01],
    })
    return buses, lines


def ieee14_data():
    """Sistema IEEE de 14 nodos (versión representativa)."""
    buses = pd.DataFrame({
        'bus_id': list(range(1, 15)),
        'type':   ['slack','PV','PQ','PV','PQ','PQ','PQ','PQ',
                   'PQ','PQ','PQ','PQ','PQ','PQ'],
        'V_mag':  [1.060,1.045,1.010,1.020]+[1.0]*10,
        'V_ang':  [0.0]*14,
        'P_gen':  [2.324,0.40,0,0]+[0]*10,
        'Q_gen':  [0.0]*14,
        'P_load': [0,0.217,0.942,0.478,0.076,0.112,0,0,
                   0.295,0.09,0.035,0.061,0.135,0.149],
        'Q_load': [0,0.127,0.19,0.039,0.016,0.075,0,0,
                   0.166,0.058,0.018,0.016,0.058,0.05],
    })
    lines = pd.DataFrame({
        'from_bus':[1,1,2,2,2,3,4,4,5,6,6,6,7,7,9,9,10,12,13],
        'to_bus':  [2,5,3,4,5,4,5,7,6,11,12,13,8,9,10,14,11,13,14],
        'R':       [0.01938,0.05403,0.04699,0.05811,0.05695,0.06701,
                    0.01335,0,0,0.09498,0.12291,0.06615,0,0,
                    0.03181,0.12711,0.08205,0.22092,0.17093],
        'X':       [0.05917,0.22304,0.19797,0.17632,0.17388,0.17103,
                    0.04211,0.20912,0.55618,0.1989,0.25581,0.13027,
                    0.17615,0.11001,0.0845,0.27038,0.19207,0.19988,0.34802],
        'B_total': [0.0528,0.0492,0.0438,0.0374,0.034,0.0346,
                    0.0128,0,0,0,0,0,0,0,0,0,0,0,0],
    })
    return buses, lines


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(bus_data, line_data, system_name="Sistema",
        S_base=100.0, tol=1e-6, max_iter=50, load_mult=1.0,
        save_plot=None) -> PowerFlowResult:
    """
    Pipeline completo:
      1. Construir Ybus
      2. Escalar cargas
      3. Resolver Newton-Raphson
      4. Imprimir y graficar resultados
    """
    # Escalar cargas si load_mult ≠ 1.0
    bd = bus_data.copy()
    bd['P_load'] = bd['P_load'] * load_mult
    bd['Q_load'] = bd['Q_load'] * load_mult

    print(f"\n{'█'*62}")
    print(f"  {system_name}  |  S_base={S_base} MVA  |  Carga×{load_mult:.2f}")
    print(f"{'█'*62}")

    print("\n[1/3] Construyendo Ybus...")
    Ybus = YbusBuilder.build(bd, line_data)
    print(f"  ✓ Ybus {Ybus.shape[0]}×{Ybus.shape[1]}  "
          f"κ = {np.linalg.cond(Ybus):.3e}")

    print("\n[2/3] Resolviendo flujo de potencia...")
    solver = PowerFlowSolver(tol=tol, max_iter=max_iter,
                             S_base_MVA=S_base, verbose=True)
    result = solver.solve(bd, line_data, Ybus)

    print("[3/3] Resultados:\n")
    if result.bus_results is not None:
        print("  NODOS")
        print(result.bus_results.to_string(index=False))
    if result.line_results is not None:
        print("\n  LÍNEAS")
        print(result.line_results.to_string(index=False))
    print(f"\n  Pérdidas activas:   {result.total_losses_MW:.4f} MW")
    print(f"  Pérdidas reactivas: {result.total_losses_MVAr:.4f} MVAr")
    print(f"  {result.message}\n")

    plot_results(result, system_name=system_name, save_path=save_plot)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Simulador de Flujo de Potencia — Newton-Raphson'
    )
    parser.add_argument('--csv', nargs=2, metavar=('BUSES', 'LINES'),
                        help='Archivos CSV de entrada')
    parser.add_argument('--system', choices=['ieee5', 'ieee14'],
                        default='ieee5', help='Sistema predefinido')
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=50)
    parser.add_argument('--Sbase', type=float, default=100.0)
    parser.add_argument('--load', type=float, default=1.0,
                        help='Multiplicador de carga (ej. 1.2 = 120%%)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.csv:
        bus_data  = pd.read_csv(args.csv[0])
        line_data = pd.read_csv(args.csv[1])
        name = f"Sistema desde CSV"
    elif args.system == 'ieee14':
        bus_data, line_data = ieee14_data()
        name = "IEEE 14-Bus"
    else:
        bus_data, line_data = ieee5_data()
        name = "IEEE 5-Bus"

    result = run(bus_data, line_data, system_name=name,
                 S_base=args.Sbase, tol=args.tol, max_iter=args.maxiter,
                 load_mult=args.load, save_plot=args.output)

    return 0 if result.converged else 1


if __name__ == '__main__':
    sys.exit(main())
