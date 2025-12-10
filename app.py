import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================
#  IMPORTAR TUS FUNCIONES
# ==========================
from biseccion import bisection_iterations
from NewtonRaphson import newton_raphson_iterations
from secante import secant_iterations
from gauss_jordan import gauss_jordan
from gauss_seidel import gauss_seidel
from jacobi import jacobi
from MatrizInversa import resolver_matriz_inversa
from lagrange import lagrange_eval, lagrange_iterations
from INewton import newton_eval, newton_iterations
from MinimosCuadrados import fit_least_squares
from PuntoFijo import punto_fijo_multivariable, mostrar_historial_puntofijo

# ==========================
#  ESTILO MATPLOTLIB GLOBAL
# ==========================
plt.style.use("dark_background")


def pretty_axes(ax, x_label="x", y_label="y", title=None):
    """Aplica estilo bonito a un eje de Matplotlib."""
    ax.spines["top"].set_alpha(0.2)
    ax.spines["right"].set_alpha(0.2)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#e5e7eb")
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)


# ==========================================================
#  FUNCIONES AUXILIARES GENERALES
# ==========================================================

def crear_funcion_1var(expr: str):
    """
    Convierte un string como 'x**2 - 4' o 'math.sin(x)' en una función f(x).
    Usa math y numpy.
    """
    expr = expr.replace("^", "**")
    safe = {
        "math": math,
        "np": np,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    def f(x):
        return eval(expr, {"__builtins__": {}}, {**safe, "x": x})

    return f


def parse_lista_float(texto: str):
    """
    '1,2,3'  -> np.array([1.,2.,3.])
    '1 2 3'  -> np.array([1.,2.,3.])
    """
    texto = texto.replace(";", ",").replace(" ", ",")
    partes = [p for p in texto.split(",") if p.strip() != ""]
    return np.array([float(p) for p in partes], dtype=float)


def parse_matriz(texto: str):
    """
    Convierte una matriz escrita como:
    3,2,-1
    2,-2,4
    -1,0.5,-1
    en un np.array 2D.
    """
    filas = []
    for linea in texto.strip().splitlines():
        if not linea.strip():
            continue
        fila = parse_lista_float(linea)
        filas.append(fila)
    return np.vstack(filas)


# --------- Integración numérica sencilla ---------

def trapecio_compuesto(f, a, b, n):
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    ys = [f(x) for x in xs]
    integral = h * (0.5 * ys[0] + sum(ys[1:-1]) + 0.5 * ys[-1])
    return xs, ys, integral


def simpson_simple(f, a, b):
    m = (a + b) / 2.0
    fa, fm, fb = f(a), f(m), f(b)
    integral = (b - a) / 6.0 * (fa + 4 * fm + fb)
    xs = [a, m, b]
    ys = [fa, fm, fb]
    return xs, ys, integral


def simpson_compuesto(f, a, b, n):
    """
    Simpson 1/3 compuesto, n par.
    """
    if n % 2 != 0:
        raise ValueError("n debe ser par para Simpson 1/3 compuesto.")
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    ys = [f(x) for x in xs]
    suma_impares = sum(ys[1:-1:2])
    suma_pares = sum(ys[2:-1:2])
    integral = (h / 3.0) * (ys[0] + ys[-1] + 4 * suma_impares + 2 * suma_pares)
    return xs, ys, integral


# ==========================================================
#  CONFIGURACIÓN GENERAL + ESTILO
# ==========================================================

st.set_page_config(
    page_title="Proyecto Final - MODELADO COMPUTACIONAL PARA INGENIERÍA",
    page_icon="📐",
    layout="wide",
)

CUSTOM_CSS = """
<style>
main {
    background: radial-gradient(circle at top left, #1f2933 0, #020617 55%, #000000 100%);
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #f9fafb;
}
a {
    color: #38bdf8;
}
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(148,163,184,0.4);
}
.card {
    padding: 1rem 1.3rem;
    border-radius: 0.9rem;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.85));
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 16px 32px rgba(15,23,42,0.9);
}
.stTextInput > div > input,
.stNumberInput input,
textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Encabezado
st.markdown(
    """
<div class="card" style="margin-bottom:1.2rem;">
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
    <div style="
        width:60px;height:60px;border-radius:999px;
        background:conic-gradient(from 140deg,#22c55e,#3b82f6,#a855f7,#22c55e);
        display:flex;align-items:center;justify-content:center;
        font-size:30px;">
        📐
    </div>
    <div>
      <h1 style="margin-bottom:0.2rem;">Proyecto Final – Modelado Computacional</h1>
      <p style="margin:0;color:#9ca3af;font-size:0.95rem;">
        Colección interactiva de métodos numéricos,integracion y ecuaciones diferenciales  desarrollados en Python, llevados a la web con Streamlit.
      </p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================
#  MENÚ LATERAL POR UNIDADES
# ==========================

st.sidebar.title("Navegación")

unidad = st.sidebar.radio(
    "Unidad",
    ["Primera unidad", "Segunda unidad"],
)

if unidad == "Primera unidad":
    seccion = st.sidebar.radio(
        "Temas de la primera unidad",
        [
            "Raíces de ecuaciones (1 variable)",
            "Sistemas lineales",
            "Sistemas no lineales (punto fijo multivariable)",
        ],
    )
else:  # Segunda unidad
    seccion = st.sidebar.radio(
        "Temas de la segunda unidad",
        [
            "Interpolación / Ajuste",
            "Integración numérica",
            "Ecuaciones diferenciales (EDO)",
        ],
    )

st.sidebar.markdown("---")

# Mensaje descriptivo según la sección elegida
if seccion == "Raíces de ecuaciones (1 variable)":
    st.sidebar.write("🔍 Métodos para encontrar raíces en ℝ: Bisección, Newton y Secante.")
elif seccion == "Sistemas lineales":
    st.sidebar.write("🧊 Resolución de Ax = b con métodos directos e iterativos.")
elif seccion == "Sistemas no lineales (punto fijo multivariable)":
    st.sidebar.write("🧷 Resolución de sistemas no lineales mediante esquemas de punto fijo.")
elif seccion == "Interpolación / Ajuste":
    st.sidebar.write("📈 Reconstrucción y ajuste de datos con polinomios.")
elif seccion == "Integración numérica":
    st.sidebar.write("∫ Cálculo aproximado de áreas bajo la curva.")
elif seccion == "Ecuaciones diferenciales (EDO)":
    st.sidebar.write("📚 Resolución numérica de EDO de 1er orden (Euler, Taylor, Runge–Kutta).")

st.sidebar.info(
    "Autor: **Grupo 8**\nProyecto Final.",
    icon="👨‍💻",
)


# ==========================================================
#  1) RAÍCES – BISECCIÓN, NEWTON, SECANTE
# ==========================================================

if seccion == "Raíces de ecuaciones (1 variable)":
    st.subheader("🔍 Cálculo de raíces en ℝ")

    tab_bis, tab_newton, tab_secante = st.tabs(
        ["⚖ Bisección", "🧮 Newton-Raphson", "📉 Secante"]
    )

    # ----- BISECCIÓN -----
    with tab_bis:
        col1, col2 = st.columns([1.1, 1.2])

        with col1:
            st.markdown("#### Parámetros")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            expr = st.text_input("f(x) =", "x**3 - x - 2", key="bis_expr")
            a = st.number_input("Extremo izquierdo a", value=1.0, key="bis_a")
            b = st.number_input("Extremo derecho b", value=2.0, key="bis_b")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="bis_tol")
            maxit = st.number_input("Máx. iteraciones", value=50, step=1, key="bis_maxit")
            rango_plot = st.slider(
                "Rango extra para la gráfica alrededor del intervalo",
                0.0, 5.0, 1.0, 0.5,
                key="bis_rango"
            )
            ejecutar_bis = st.button("Calcular Bisección")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Salida")
            st.markdown('<div class="card">', unsafe_allow_html=True)

            if ejecutar_bis:
                try:
                    f = crear_funcion_1var(expr)
                    filas = bisection_iterations(f, a, b, tol, int(maxit))
                    df = pd.DataFrame(
                        filas,
                        columns=["k", "a_k", "b_k", "r_k", "f(r_k)", "Error"],
                    )
                    r_final = df["r_k"].iloc[-1]
                    err_final = df["Error"].iloc[-1]

                    c1, c2 = st.columns(2)
                    c1.metric("Raíz aproximada", f"{r_final:.8f}")
                    c2.metric("Error final", f"{err_final:.2e}")

                    tab_graf, tab_tabla = st.tabs(["📈 Gráfica", "📋 Tabla de iteraciones"])

                    with tab_tabla:
                        with st.expander("Ver tabla completa de iteraciones", expanded=True):
                            st.dataframe(df, use_container_width=True, height=260)

                    with tab_graf:
                        x_min = min(a, r_final) - rango_plot
                        x_max = max(b, r_final) + rango_plot
                        xs = np.linspace(x_min, x_max, 400)
                        ys = [f(x) for x in xs]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.axhline(0, color="#f97316", linewidth=1.1, alpha=0.8)
                        ax.plot(xs, ys, linewidth=2, label="f(x)")
                        ax.scatter([r_final], [f(r_final)], color="#22c55e", s=60,
                                   zorder=5, label="Raíz aprox.")
                        ax.vlines(r_final, 0, f(r_final), colors="#22c55e",
                                  linestyles="--", alpha=0.7)
                        pretty_axes(ax, title="f(x) y raíz por Bisección")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura los parámetros y presiona **Calcular Bisección**.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ----- NEWTON-RAPHSON -----
    with tab_newton:
        col1, col2 = st.columns([1.1, 1.2])

        with col1:
            st.markdown("#### Parámetros")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            expr_n = st.text_input("f(x) =", "x**3 - x - 2", key="new_expr")
            expr_df = st.text_input("f'(x) =", "3*x**2 - 1", key="new_expr_df")
            x0 = st.number_input("x₀", value=2.0, key="new_x0")
            tol_n = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="new_tol")
            maxit_n = st.number_input("Máx. iteraciones", value=50, step=1, key="new_maxit")
            rango_plot_n = st.slider(
                "Rango alrededor de la raíz para graficar",
                1.0, 10.0, 3.0, 0.5,
                key="new_rango"
            )
            ejecutar_new = st.button("Calcular Newton-Raphson")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Salida")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if ejecutar_new:
                try:
                    f = crear_funcion_1var(expr_n)
                    df_expr = crear_funcion_1var(expr_df)
                    filas = newton_raphson_iterations(
                        f, df_expr, float(x0), tol_n, int(maxit_n)
                    )
                    df_table = pd.DataFrame(
                        filas,
                        columns=["k", "x_k", "f(x_k)", "f'(x_k)", "x_{k+1}", "Error"],
                    )
                    x_final = df_table["x_{k+1}"].iloc[-1]
                    err_final = df_table["Error"].iloc[-1]

                    c1, c2 = st.columns(2)
                    c1.metric("Raíz aproximada", f"{x_final:.8f}")
                    c2.metric("Error final", f"{err_final:.2e}")

                    tab_graf, tab_tabla = st.tabs(["📈 Gráfica", "📋 Tabla"])

                    with tab_tabla:
                        with st.expander("Ver tabla de iteraciones", expanded=True):
                            st.dataframe(df_table, use_container_width=True, height=260)

                    with tab_graf:
                        xs = np.linspace(x_final - rango_plot_n, x_final + rango_plot_n, 400)
                        ys = [f(x) for x in xs]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.axhline(0, color="#f97316", linewidth=1.1, alpha=0.8)
                        ax.plot(xs, ys, linewidth=2, label="f(x)")
                        ax.scatter([x_final], [0], color="#22c55e", s=70, label="Raíz aprox.")
                        pretty_axes(ax, title="Newton-Raphson")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura y presiona **Calcular Newton-Raphson**.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ----- SECANTE -----
    with tab_secante:
        col1, col2 = st.columns([1.1, 1.2])

        with col1:
            st.markdown("#### Parámetros")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            expr_s = st.text_input("f(x) =", "x**3 - x - 2", key="sec_expr")
            x0_s = st.number_input("x₀", value=1.0, key="sec_x0")
            x1_s = st.number_input("x₁", value=2.0, key="sec_x1")
            tol_s = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="sec_tol")
            maxit_s = st.number_input("Máx. iteraciones", value=50, step=1, key="sec_maxit")
            rango_plot_s = st.slider(
                "Rango alrededor de la raíz para graficar",
                1.0, 10.0, 3.0, 0.5,
                key="sec_rango"
            )
            ejecutar_sec = st.button("Calcular Secante")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Salida")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if ejecutar_sec:
                try:
                    f = crear_funcion_1var(expr_s)
                    filas = secant_iterations(f, float(x0_s), float(x1_s), tol_s, int(maxit_s))
                    df_table = pd.DataFrame(
                        filas,
                        columns=["k", "x0", "x1", "x2", "f(x2)", "Error"],
                    )
                    x_final = df_table["x2"].iloc[-1]
                    err_final = df_table["Error"].iloc[-1]

                    c1, c2 = st.columns(2)
                    c1.metric("Raíz aproximada", f"{x_final:.8f}")
                    c2.metric("Error final", f"{err_final:.2e}")

                    tab_graf, tab_tabla = st.tabs(["📈 Gráfica", "📋 Tabla"])

                    with tab_tabla:
                        with st.expander("Ver tabla de iteraciones", expanded=True):
                            st.dataframe(df_table, use_container_width=True, height=260)

                    with tab_graf:
                        xs = np.linspace(x_final - rango_plot_s, x_final + rango_plot_s, 400)
                        ys = [f(x) for x in xs]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.axhline(0, color="#f97316", linewidth=1.1, alpha=0.8)
                        ax.plot(xs, ys, linewidth=2, label="f(x)")
                        ax.scatter([x_final], [0], color="#22c55e", s=70, label="Raíz aprox.")
                        pretty_axes(ax, title="Método de la Secante")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura y presiona **Calcular Secante**.")
            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
#  2) SISTEMAS LINEALES
# ==========================================================

elif seccion == "Sistemas lineales":
    st.subheader("🧊 Sistemas de ecuaciones lineales")

    metodo = st.radio(
        "Selecciona método",
        ["Gauss-Jordan", "Jacobi", "Gauss-Seidel", "Matriz inversa"],
        horizontal=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    colA, colB = st.columns([2, 1])

    ejemplo_A = "3,2,-1\n2,-2,4\n-1,0.5,-1"
    ejemplo_b = "1,-2,0"

    with colA:
        texto_A = st.text_area("Matriz A (cada fila en una línea)", ejemplo_A, height=170)
    with colB:
        texto_b = st.text_input("Vector b", ejemplo_b)

        tol = 1e-6
        maxit = 50
        if metodo in ["Jacobi", "Gauss-Seidel"]:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
            maxit = st.number_input("Máx. iteraciones", value=50, step=1)

        ejecutar_sis = st.button("Resolver sistema")

    st.markdown("</div>", unsafe_allow_html=True)

    if ejecutar_sis:
        try:
            A = parse_matriz(texto_A)
            b = parse_lista_float(texto_b)
            if A.shape[0] != A.shape[1]:
                raise ValueError("A debe ser cuadrada.")
            if b.size != A.shape[0]:
                raise ValueError("El tamaño de b no coincide con A.")

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if metodo == "Gauss-Jordan":
                x, M_final = gauss_jordan(A.copy(), b.copy())
                st.markdown("#### Solución")
                for i, xi in enumerate(x):
                    st.write(f"x{i+1} = {xi:.8f}")
                with st.expander("Ver matriz aumentada final [A|b]"):
                    st.dataframe(pd.DataFrame(M_final), use_container_width=True)

            elif metodo == "Jacobi":
                iteraciones, x_final = jacobi(A.copy(), b.copy(), tol, int(maxit))
                cols = ["Iteración"] + [f"x{i+1}" for i in range(len(x_final))] + ["Error"]
                df = pd.DataFrame(iteraciones, columns=cols)
                st.success(f"Solución aproximada: {x_final}")
                tab_t, tab_g = st.tabs(["📋 Tabla de iteraciones", "📈 Gráfica de error"])
                with tab_t:
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.semilogy(df["Iteración"], df["Error"], marker="o")
                    pretty_axes(ax, "Iteración", "Error (∞-norma)", "Convergencia Jacobi")
                    st.pyplot(fig)

            elif metodo == "Gauss-Seidel":
                iteraciones, x_final = gauss_seidel(A.copy(), b.copy(), tol, int(maxit))
                cols = ["Iteración"] + [f"x{i+1}" for i in range(len(x_final))] + ["Error"]
                df = pd.DataFrame(iteraciones, columns=cols)
                st.success(f"Solución aproximada: {x_final}")
                tab_t, tab_g = st.tabs(["📋 Tabla de iteraciones", "📈 Gráfica de error"])
                with tab_t:
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.semilogy(df["Iteración"], df["Error"], marker="o")
                    pretty_axes(ax, "Iteración", "Error (∞-norma)", "Convergencia Gauss-Seidel")
                    st.pyplot(fig)

            else:  # Matriz inversa
                x, A_inv, det = resolver_matriz_inversa(A.copy(), b.copy())
                st.markdown("#### Solución")
                for i, xi in enumerate(x):
                    st.write(f"x{i+1} = {xi:.8f}")
                st.info(f"Determinante det(A) = {det:.6e}")
                with st.expander("Ver matriz inversa A⁻¹"):
                    st.dataframe(pd.DataFrame(A_inv), use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")


# ==========================================================
#  3) INTERPOLACIÓN / AJUSTE
# ==========================================================

elif seccion == "Interpolación / Ajuste":
    st.subheader("📈 Interpolación y ajuste de datos")

    tab_lag, tab_newt, tab_ls = st.tabs(
        ["📐 Lagrange", "📐 Newton (dif. divididas)", "📊 Mínimos cuadrados"]
    )

    # Lagrange
    with tab_lag:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        x_text = st.text_input("X (comas)", "1,2,3,4", key="lag_x")
        y_text = st.text_input("Y (comas)", "1,4,9,16", key="lag_y")
        x_eval = st.number_input("x a evaluar", value=2.5, key="lag_xeval")
        ejecutar_lag = st.button("Calcular Lagrange")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_lag:
            try:
                xs = parse_lista_float(x_text)
                ys = parse_lista_float(y_text)
                if xs.size != ys.size:
                    raise ValueError("X e Y deben tener la misma cantidad de puntos.")

                filas = lagrange_iterations(xs, ys, x_eval)
                resultado = lagrange_eval(x_eval, xs, ys)
                df = pd.DataFrame(
                    filas,
                    columns=["Punto", "xi", "yi", "Li", "yi*Li"],
                )

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**P({x_eval}) ≈ {resultado:.6f}**")

                tab_t, tab_g = st.tabs(["📋 Tabla de términos", "📈 Gráfica"])
                with tab_t:
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    x_min, x_max = xs.min() - 1, xs.max() + 1
                    x_plot = np.linspace(x_min, x_max, 300)
                    y_plot = [lagrange_eval(x, xs, ys) for x in x_plot]

                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(x_plot, y_plot, linewidth=2, label="Polinomio de Lagrange")
                    ax.scatter(xs, ys, color="#f97316", label="Datos", zorder=5)
                    ax.scatter([x_eval], [resultado], color="#22c55e", label=f"P({x_eval})")
                    pretty_axes(ax, title="Interpolación de Lagrange")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

    # Newton diferencias divididas
    with tab_newt:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        x_text = st.text_input("X (comas)", "1,2,3,4", key="newt_x")
        y_text = st.text_input("Y (comas)", "1,4,9,16", key="newt_y")
        x_eval = st.number_input("x a evaluar", value=2.5, key="newt_xeval")
        ejecutar_newt = st.button("Calcular Newton (dif. divididas)")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_newt:
            try:
                xs = parse_lista_float(x_text)
                ys = parse_lista_float(y_text)
                if xs.size != ys.size:
                    raise ValueError("X e Y deben tener la misma cantidad de puntos.")

                filas, coef, tabla = newton_iterations(xs, ys, x_eval)
                resultado = newton_eval(coef, xs, x_eval)
                df = pd.DataFrame(filas, columns=["i", "xi", "yi"])

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**P({x_eval}) ≈ {resultado:.6f}**")

                tab_t, tab_g = st.tabs(["📋 Datos", "📈 Gráfica"])
                with tab_t:
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    x_min, x_max = xs.min() - 1, xs.max() + 1
                    x_plot = np.linspace(x_min, x_max, 300)
                    y_plot = [newton_eval(coef, xs, x) for x in x_plot]

                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(x_plot, y_plot, linewidth=2, label="Polinomio de Newton")
                    ax.scatter(xs, ys, color="#f97316", label="Datos", zorder=5)
                    ax.scatter([x_eval], [resultado], color="#22c55e", label=f"P({x_eval})")
                    pretty_axes(ax, title="Interpolación de Newton (dif. divididas)")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

    # Mínimos cuadrados
    with tab_ls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        x_text = st.text_input("X (comas)", "1,2,3,4,5", key="ls_x")
        y_text = st.text_input("Y (comas)", "2.1,3.9,6.2,7.8,10.1", key="ls_y")
        grado = st.number_input("Grado del polinomio", value=1, min_value=1, max_value=10)
        ejecutar_ls = st.button("Calcular ajuste por mínimos cuadrados")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_ls:
            try:
                xs = parse_lista_float(x_text)
                ys = parse_lista_float(y_text)
                if xs.size != ys.size:
                    raise ValueError("X e Y deben tener la misma cantidad de puntos.")

                coefs, poly, ys_pred, errors, rmse, sse = fit_least_squares(
                    xs, ys, int(grado)
                )

                st.markdown('<div class="card">', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("RMSE", f"{rmse:.6f}")
                c2.metric("SSE", f"{sse:.6f}")
                st.write(f"**Coeficientes:** {coefs}")

                df = pd.DataFrame(
                    {
                        "x": xs,
                        "y": ys,
                        "y_pred": ys_pred,
                        "error": errors,
                    }
                )

                tab_t, tab_g = st.tabs(["📋 Tabla de datos", "📈 Gráfica de ajuste"])
                with tab_t:
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    x_min, x_max = xs.min() - 1, xs.max() + 1
                    x_plot = np.linspace(x_min, x_max, 300)
                    y_plot = poly(x_plot)

                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.scatter(xs, ys, label="Datos", zorder=5)
                    ax.plot(x_plot, y_plot, label="Ajuste", linewidth=2)
                    pretty_axes(ax, title="Ajuste polinomial por mínimos cuadrados")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")


# ==========================================================
#  4) INTEGRACIÓN NUMÉRICA
# ==========================================================

elif seccion == "Integración numérica":
    st.subheader("∫ Integración numérica")

    tab_trap, tab_simp_s, tab_simp_c = st.tabs(
        ["🔺 Trapecio compuesto", "📐 Simpson 1/3 simple", "📐 Simpson 1/3 compuesto"]
    )

    # Trapecio
    with tab_trap:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        expr = st.text_input("f(x) =", "math.sin(x) + x**2", key="trap_expr")
        a = st.number_input("Límite inferior a", value=0.0, key="trap_a")
        b = st.number_input("Límite superior b", value=3.1416, key="trap_b")
        n = st.number_input("Número de subintervalos n", value=4, step=1, key="trap_n")
        ejecutar_trap = st.button("Calcular Trapecio compuesto")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_trap:
            try:
                f = crear_funcion_1var(expr)
                xs, ys, I = trapecio_compuesto(f, a, b, int(n))

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Integral aproximada", f"{I:.10f}")

                tab_t, tab_g = st.tabs(["📋 Tabla de puntos", "📈 Área aproximada"])
                with tab_t:
                    df = pd.DataFrame({"x": xs, "f(x)": ys})
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(xs, ys, marker="o", label="f(x)")
                    ax.fill_between(xs, ys, alpha=0.4, label="Área (trapecios)")
                    ax.axhline(0, color="#f97316", linewidth=1.0, alpha=0.8)
                    pretty_axes(ax, title="Método del Trapecio compuesto")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

    # Simpson simple
    with tab_simp_s:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        expr = st.text_input("f(x) =", "math.sin(x) + x**2", key="simp_s_expr")
        a = st.number_input("Límite inferior a", value=0.0, key="simp_s_a")
        b = st.number_input("Límite superior b", value=3.1416, key="simp_s_b")
        ejecutar_simp_s = st.button("Calcular Simpson 1/3 simple")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_simp_s:
            try:
                f = crear_funcion_1var(expr)
                xs, ys, I = simpson_simple(f, a, b)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Integral aproximada", f"{I:.10f}")

                tab_t, tab_g = st.tabs(["📋 Puntos usados", "📈 Área aproximada"])
                with tab_t:
                    df = pd.DataFrame({"x": xs, "f(x)": ys})
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(xs, ys, marker="o", label="f(x)")
                    ax.fill_between(xs, ys, alpha=0.4, label="Área (Simpson 1/3)")
                    ax.axhline(0, color="#f97316", linewidth=1.0, alpha=0.8)
                    pretty_axes(ax, title="Método de Simpson 1/3 simple")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

    # Simpson compuesto
    with tab_simp_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        expr = st.text_input("f(x) =", "math.sin(x) + x**2", key="simp_c_expr")
        a = st.number_input("Límite inferior a", value=0.0, key="simp_c_a")
        b = st.number_input("Límite superior b", value=3.1416, key="simp_c_b")
        n = st.number_input("Número de subintervalos n (par)", value=4, step=2, key="simp_c_n")
        ejecutar_simp_c = st.button("Calcular Simpson 1/3 compuesto")
        st.markdown("</div>", unsafe_allow_html=True)

        if ejecutar_simp_c:
            try:
                f = crear_funcion_1var(expr)
                xs, ys, I = simpson_compuesto(f, a, b, int(n))

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Integral aproximada", f"{I:.10f}")

                tab_t, tab_g = st.tabs(["📋 Puntos usados", "📈 Área aproximada"])
                with tab_t:
                    df = pd.DataFrame({"x": xs, "f(x)": ys})
                    st.dataframe(df, use_container_width=True, height=260)
                with tab_g:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(xs, ys, marker="o", label="f(x)")
                    ax.fill_between(xs, ys, alpha=0.4, label="Área (Simpson 1/3 comp.)")
                    ax.axhline(0, color="#f97316", linewidth=1.0, alpha=0.8)
                    pretty_axes(ax, title="Método de Simpson 1/3 compuesto")
                    ax.legend()
                    st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")


# ==========================================================
#  5) ECUACIONES DIFERENCIALES (EDO)
# ==========================================================

elif seccion == "Ecuaciones diferenciales (EDO)":
    st.subheader("📚 Ecuaciones diferenciales de 1er orden (EDO)")

    # --------- FUNCIONES AUXILIARES SOLO PARA ESTA SECCIÓN ---------
    SAFE_MATH = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "pow": pow,
    }

    def crear_funcion_2var(expr: str):
        """
        f(t, y) desde un string como 'y - t**2 + 1' o 't + y'.
        Se permiten t, x (como sinónimo de t) y y.
        """
        expr = expr.replace("^", "**").strip()

        def f(t, y):
            local = {"t": t, "x": t, "y": y}
            local.update(SAFE_MATH)
            return eval(expr, {"__builtins__": None}, local)

        return f

    def crear_funcion_1var_t(expr: str):
        """
        g(t) para solución exacta opcional. Se permiten t o x.
        """
        expr = expr.replace("^", "**").strip()

        def g(t):
            local = {"t": t, "x": t}
            local.update(SAFE_MATH)
            return eval(expr, {"__builtins__": None}, local)

        return g

    def crear_funcion_ypp(expr: str):
        """
        y''(t, y, yp) para Taylor 2º orden. Variables: t/x, y, yp.
        """
        expr = expr.replace("^", "**").strip()

        def ypp(t, y, yp):
            local = {"t": t, "x": t, "y": y, "yp": yp}
            local.update(SAFE_MATH)
            return eval(expr, {"__builtins__": None}, local)

        return ypp

    # --------- MÉTODOS NUMÉRICOS ---------

    def euler_method(f, t0, y0, h, steps):
        ts = [t0]
        ys = [y0]
        t, y = t0, y0
        for _ in range(steps):
            y = y + h * f(t, y)
            t = t + h
            ts.append(round(t, 12))
            ys.append(y)
        return ts, ys

    def taylor2_method(f, ypp_func, t0, y0, h, steps):
        ts = [t0]
        ys = [y0]
        t, y = t0, y0
        for _ in range(steps):
            yp = f(t, y)
            ypp = ypp_func(t, y, yp)
            y = y + h * yp + (h**2 / 2.0) * ypp
            t = t + h
            ts.append(round(t, 12))
            ys.append(y)
        return ts, ys

    def rk2_heun(f, t0, y0, h, steps):
        """
        Runge–Kutta 2 (Heun).
        """
        ts = [t0]
        ys = [y0]
        t, y = t0, y0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h, y + h * k1)
            y = y + (h / 2.0) * (k1 + k2)
            t = t + h
            ts.append(round(t, 12))
            ys.append(y)
        return ts, ys

    def rk4(f, t0, y0, h, steps):
        ts = [t0]
        ys = [y0]
        t, y = t0, y0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + (h / 2.0) * k1)
            k3 = f(t + h / 2.0, y + (h / 2.0) * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + h
            ts.append(round(t, 12))
            ys.append(y)
        return ts, ys

    # --------- INTERFAZ CON TABS ---------

    tab_euler, tab_taylor, tab_rk = st.tabs(
        ["🧮 Método de Euler", "📐 Taylor 2° orden", "🚀 Runge–Kutta (RK2 / RK4)"]
    )

    # ===== EULER =====
    with tab_euler:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 1.4])

        with col1:
            st.markdown("#### Parámetros – Euler")
            expr_f = st.text_input(
                "y' = f(t, y)",
                "y - t**2 + 1",
                key="eul_f",
                help="Puedes usar t, y, sin, cos, exp, sqrt, etc.",
            )
            expr_exact = st.text_input(
                "Solución exacta y(t) (opcional)",
                "",
                key="eul_exact",
                help="Ejemplo: exp(t) + t**2 - 1. Déjalo vacío si no la conoces.",
            )
            t0 = st.number_input("t₀", value=0.0, key="eul_t0")
            y0 = st.number_input("y₀", value=0.5, key="eul_y0")
            h = st.number_input("Paso h", value=0.1, key="eul_h")
            steps = st.number_input("Número de pasos", value=5, step=1, min_value=1, key="eul_steps")
            ejecutar_euler = st.button("Calcular Euler", key="btn_euler")

        with col2:
            st.markdown("#### Resultado – Euler")
            if ejecutar_euler:
                try:
                    f = crear_funcion_2var(expr_f)
                    ts, ys = euler_method(f, t0, y0, h, int(steps))

                    # Solución exacta si se dio
                    y_exact = None
                    if expr_exact.strip() != "":
                        g = crear_funcion_1var_t(expr_exact)
                        y_exact = [g(t) for t in ts]

                    c1, c2 = st.columns(2)
                    c1.metric("t final", f"{ts[-1]:.6f}")
                    c2.metric("y(t final)", f"{ys[-1]:.6f}")

                    # Tabla
                    df_dict = {"t": ts, "y_euler": ys}
                    if y_exact is not None:
                        df_dict["y_exacta"] = y_exact
                        df_dict["error"] = [y_exact[i] - ys[i] for i in range(len(ts))]
                    df = pd.DataFrame(df_dict)

                    tab_tabla, tab_graf = st.tabs(["📋 Tabla", "📈 Gráfica"])
                    with tab_tabla:
                        st.dataframe(df, use_container_width=True, height=260)

                    with tab_graf:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(ts, ys, "o--", label="Euler", linewidth=1.5)
                        if y_exact is not None:
                            ax.plot(ts, y_exact, "-", label="Exacta", linewidth=2)
                        pretty_axes(ax, "t", "y", "Método de Euler")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura los parámetros y presiona **Calcular Euler**.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== TAYLOR 2º ORDEN =====
    with tab_taylor:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 1.4])

        with col1:
            st.markdown("#### Parámetros – Taylor 2°")
            expr_f_t = st.text_input(
                "y' = f(t, y)",
                "t + y",
                key="tay_f",
                help="Derivada primera f(t,y).",
            )
            expr_ypp = st.text_input(
                "y'' = y''(t, y, yp)",
                "1 + t + y",
                key="tay_ypp",
                help="Puedes usar t, y, yp.",
            )
            expr_exact_t = st.text_input(
                "Solución exacta y(t) (opcional)",
                "",
                key="tay_exact",
            )
            t0_t = st.number_input("t₀", value=0.0, key="tay_t0")
            y0_t = st.number_input("y₀", value=1.0, key="tay_y0")
            h_t = st.number_input("Paso h", value=0.1, key="tay_h")
            steps_t = st.number_input("Número de pasos", value=5, step=1, min_value=1, key="tay_steps")
            ejecutar_taylor = st.button("Calcular Taylor 2°", key="btn_taylor")

        with col2:
            st.markdown("#### Resultado – Taylor 2°")
            if ejecutar_taylor:
                try:
                    f_t = crear_funcion_2var(expr_f_t)
                    ypp_func = crear_funcion_ypp(expr_ypp)
                    ts, ys = taylor2_method(f_t, ypp_func, t0_t, y0_t, h_t, int(steps_t))

                    y_exact_t = None
                    if expr_exact_t.strip() != "":
                        g_t = crear_funcion_1var_t(expr_exact_t)
                        y_exact_t = [g_t(t) for t in ts]

                    c1, c2 = st.columns(2)
                    c1.metric("t final", f"{ts[-1]:.6f}")
                    c2.metric("y(t final)", f"{ys[-1]:.6f}")

                    df_dict = {"t": ts, "y_taylor2": ys}
                    if y_exact_t is not None:
                        df_dict["y_exacta"] = y_exact_t
                        df_dict["error"] = [y_exact_t[i] - ys[i] for i in range(len(ts))]
                    df = pd.DataFrame(df_dict)

                    tab_tabla, tab_graf = st.tabs(["📋 Tabla", "📈 Gráfica"])
                    with tab_tabla:
                        st.dataframe(df, use_container_width=True, height=260)
                    with tab_graf:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(ts, ys, "o--", label="Taylor 2°", linewidth=1.5)
                        if y_exact_t is not None:
                            ax.plot(ts, y_exact_t, "-", label="Exacta", linewidth=2)
                        pretty_axes(ax, "t", "y", "Método de Taylor 2° orden")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura los parámetros y presiona **Calcular Taylor 2°**.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== RUNGE–KUTTA (RK2 / RK4) =====
    with tab_rk:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 1.4])

        with col1:
            st.markdown("#### Parámetros – Runge–Kutta")
            expr_f_rk = st.text_input(
                "y' = f(t, y)",
                "y - t**2 + 1",
                key="rk_f",
            )
            expr_exact_rk = st.text_input(
                "Solución exacta y(t) (opcional)",
                "",
                key="rk_exact",
            )
            t0_rk = st.number_input("t₀", value=0.0, key="rk_t0")
            y0_rk = st.number_input("y₀", value=0.5, key="rk_y0")
            h_rk = st.number_input("Paso h", value=0.1, key="rk_h")
            steps_rk = st.number_input("Número de pasos", value=5, step=1, min_value=1, key="rk_steps")
            ejecutar_rk = st.button("Calcular RK2 / RK4", key="btn_rk")

        with col2:
            st.markdown("#### Resultado – Runge–Kutta")
            if ejecutar_rk:
                try:
                    f_rk = crear_funcion_2var(expr_f_rk)
                    ts2, ys2 = rk2_heun(f_rk, t0_rk, y0_rk, h_rk, int(steps_rk))
                    ts4, ys4 = rk4(f_rk, t0_rk, y0_rk, h_rk, int(steps_rk))

                    y_exact_rk = None
                    if expr_exact_rk.strip() != "":
                        g_rk = crear_funcion_1var_t(expr_exact_rk)
                        y_exact_rk = [g_rk(t) for t in ts2]  # ts2 y ts4 son iguales

                    c1, c2, c3 = st.columns(3)
                    c1.metric("t final", f"{ts2[-1]:.6f}")
                    c2.metric("y_RK2(t final)", f"{ys2[-1]:.6f}")
                    c3.metric("y_RK4(t final)", f"{ys4[-1]:.6f}")

                    df_dict = {"t": ts2, "y_RK2": ys2, "y_RK4": ys4}
                    if y_exact_rk is not None:
                        df_dict["y_exacta"] = y_exact_rk
                        df_dict["error_RK2"] = [y_exact_rk[i] - ys2[i] for i in range(len(ts2))]
                        df_dict["error_RK4"] = [y_exact_rk[i] - ys4[i] for i in range(len(ts2))]
                    df = pd.DataFrame(df_dict)

                    tab_tabla, tab_graf = st.tabs(["📋 Tabla", "📈 Gráfica"])
                    with tab_tabla:
                        st.dataframe(df, use_container_width=True, height=260)
                    with tab_graf:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(ts2, ys2, "o--", label="RK2 (Heun)", linewidth=1.5)
                        ax.plot(ts4, ys4, "s--", label="RK4", linewidth=1.5)
                        if y_exact_rk is not None:
                            ax.plot(ts2, y_exact_rk, "-", label="Exacta", linewidth=2)
                        pretty_axes(ax, "t", "y", "Runge–Kutta 2 y 4")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            else:
                st.caption("Configura los parámetros y presiona **Calcular RK2 / RK4**.")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
#  6) SISTEMAS NO LINEALES – PUNTO FIJO MULTIVARIABLE
# ==========================================================

elif seccion == "Sistemas no lineales (punto fijo multivariable)":
    st.subheader("🧷 Punto fijo multivariable")

    st.markdown(
        "Escribe la lista de funciones **g(x)** tal como en tu código original.\n\n"
        'Ejemplo: `["cos(x[1]*x[2])", "sqrt(x[0]**2 + x[2]**2)", "sin(x[0] + x[1])"]`'
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    funciones_str = st.text_area(
        "Funciones g(x) (lista de strings)",
        '["cos(x[1]*x[2])", "sqrt(x[0]**2 + x[2]**2)", "sin(x[0] + x[1])"]',
        height=120,
    )
    x0_str = st.text_input("Vector inicial x0", "[0.5, 0.5, 0.5]")
    tol = st.number_input("Tolerancia", value=1e-8, format="%.1e")
    maxit = st.number_input("Máx. iteraciones", value=50, step=1)
    ejecutar_pf = st.button("Ejecutar punto fijo")
    st.markdown("</div>", unsafe_allow_html=True)

    if ejecutar_pf:
        try:
            funcs_list = eval(funciones_str, {"__builtins__": {}}, {})
            if not isinstance(funcs_list, (list, tuple)):
                raise ValueError("Debe ser una lista de expresiones de texto.")

            def g_func(x_vec):
                local = {
                    "x": x_vec,
                    "math": math,
                    "np": np,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "sqrt": math.sqrt,
                    "exp": math.exp,
                    "log": math.log,
                }
                valores = []
                for expr in funcs_list:
                    expr2 = expr.replace("^", "**")
                    valores.append(eval(expr2, {"__builtins__": {}}, local))
                return valores

            x0 = np.array(eval(x0_str, {"__builtins__": {}}, {}), dtype=float)

            x_sol, err, k, history = punto_fijo_multivariable(
                g_func, x0, tol=tol, maxiter=int(maxit)
            )
            df_hist = mostrar_historial_puntofijo(history)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Iteraciones", f"{k}")
            c2.metric("Error final", f"{err:.2e}")

            with st.expander("Ver historial de iteraciones", expanded=True):
                st.dataframe(df_hist, use_container_width=True, height=260)

            st.success(f"Solución aproximada: {x_sol}")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

