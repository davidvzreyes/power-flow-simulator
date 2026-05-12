# ⚡ Power Flow Simulator — Newton-Raphson

Simulador interactivo de **Flujo de Potencia** en redes eléctricas, implementado con el método numérico de **Newton-Raphson**. Incluye un dashboard web interactivo y un solver en Python para uso avanzado.

---

## 🌐 Demo en vivo

> Abre `index.html` en tu navegador — no requiere instalación ni servidor.

O accede directamente en: `https://<tu-usuario>.github.io/power-flow-simulator`

---

## 📁 Estructura del proyecto

```
power-flow-simulator/
│
├── index.html              # Dashboard web interactivo (todo en uno)
├── power_flow.py           # Solver Python modular (CLI)
│
├── data/
│   ├── ieee5_buses.csv     # Nodos — Sistema IEEE 5-Bus
│   ├── ieee5_lines.csv     # Líneas — Sistema IEEE 5-Bus
│   ├── ieee14_buses.csv    # Nodos — Sistema IEEE 14-Bus
│   └── ieee14_lines.csv    # Líneas — Sistema IEEE 14-Bus
│
└── README.md
```

---

## 🖥️ Dashboard Web (`index.html`)

Abre el archivo en cualquier navegador moderno. No requiere instalar nada.

### Funcionalidades

| Función | Descripción |
|---|---|
| **Selección de sistema** | IEEE 5-Bus, IEEE 14-Bus, Mini 3-Bus |
| **Editor de nodos** | Cambia tipo (Slack/PV/PQ), voltaje de referencia, P/Q de carga y generación con sliders |
| **Multiplicador de carga** | Escala toda la demanda entre 50% y 200% para ver colapso de voltaje |
| **Tolerancia configurable** | 1e-4, 1e-6, 1e-8 |
| **Resolver automático** | Se recalcula 400ms después de cualquier cambio |
| **Pestaña Voltajes** | Perfil |V| por nodo con líneas de límites (0.95 / 1.05 pu) y ángulos θ |
| **Pestaña Flujo en líneas** | Potencia activa enviada y pérdidas por línea |
| **Pestaña Red** | Diagrama topológico con grosor de línea proporcional al flujo |
| **Pestaña Convergencia** | Log iteración a iteración del algoritmo Newton-Raphson |

### Tipos de nodos

| Tipo | Variables conocidas | Variables calculadas |
|---|---|---|
| **Slack** (referencia) | \|V\|, θ | P, Q |
| **PV** (generador) | P_gen, \|V\| | Q, θ |
| **PQ** (carga) | P_load, Q_load | \|V\|, θ |

---

## 🐍 Solver Python (`power_flow.py`)

### Requisitos

```bash
pip install numpy pandas matplotlib
```

### Uso básico

```bash
# Sistema IEEE 5-Bus (por defecto)
python power_flow.py

# Sistema IEEE 14-Bus
python power_flow.py --system ieee14

# Desde archivos CSV propios
python power_flow.py --csv data/ieee5_buses.csv data/ieee5_lines.csv

# Con parámetros personalizados
python power_flow.py --system ieee14 --tol 1e-8 --maxiter 100 --load 1.2 --output resultado.png
```

### Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--system` | `ieee5` | Sistema predefinido: `ieee5` o `ieee14` |
| `--csv BUSES LINES` | — | Archivos CSV de entrada |
| `--tol` | `1e-6` | Tolerancia de convergencia (pu) |
| `--maxiter` | `50` | Número máximo de iteraciones |
| `--Sbase` | `100.0` | Potencia base en MVA |
| `--load` | `1.0` | Multiplicador de carga (1.2 = 120%) |
| `--output` | — | Ruta para guardar la gráfica PNG |

### Formato de archivos CSV

**buses.csv**
```
bus_id,type,V_mag,V_ang,P_gen,Q_gen,P_load,Q_load
1,slack,1.06,0.0,0.0,0.0,0.0,0.0
2,PV,1.045,0.0,0.40,0.0,0.20,0.10
3,PQ,1.0,0.0,0.0,0.0,0.25,0.105
```

> Todos los valores de potencia en **pu** (S_base = 100 MVA por defecto).  
> `V_ang` en **grados**. Tipo: `slack`, `PV` o `PQ`.

**lines.csv**
```
from_bus,to_bus,R,X,B_total
1,2,0.02,0.06,0.060
1,3,0.08,0.24,0.025
```

> `R`, `X`, `B_total` en **pu**. `B_total` es la susceptancia capacitiva total de la línea (modelo π).

---

## 🔢 Base matemática

### Método Newton-Raphson

El algoritmo resuelve el sistema no-lineal de ecuaciones de balance de potencia:

```
P_i = |V_i| · Σ_j |V_j| · [G_ij·cos(θ_i-θ_j) + B_ij·sin(θ_i-θ_j)]
Q_i = |V_i| · Σ_j |V_j| · [G_ij·sin(θ_i-θ_j) - B_ij·cos(θ_i-θ_j)]
```

En cada iteración se resuelve el sistema lineal:

```
J · [Δθ, Δ|V|]ᵀ = [ΔP, ΔQ]ᵀ
```

Donde **J** es la Jacobiana particionada en 4 bloques:

```
J = | ∂P/∂θ    ∂P/∂|V| |  =  | J1  J2 |
    | ∂Q/∂θ    ∂Q/∂|V| |     | J3  J4 |
```

### Matriz de Admitancia (Ybus)

Construida con el **modelo π** de líneas de transmisión:

```
Y_ii = Σ (y_serie + y_shunt)    ← diagonal
Y_ij = -y_serie                  ← fuera de diagonal
```

donde `y_serie = 1/(R+jX)` y `y_shunt = jB/2`.

---

## 📚 Referencias

- Tinney, W.F. & Hart, C.E. (1967). *Power flow solution by Newton's method*. IEEE Trans. Power App. Syst.
- Glover, Sarma & Overbye. *Power System Analysis and Design*, 5th ed.
- Bergen & Vittal. *Power Systems Analysis*, 2nd ed.
- Sereeter, B., Vuik, C. & Witteveen, C. (2019). *On a comparison of Newton–Raphson solvers for power flow problems*. Journal of Computational and Applied Mathematics, 360, 157–169.

---

## 📄 Licencia

MIT License — libre para uso académico y comercial.
