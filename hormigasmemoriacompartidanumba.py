import time
import numpy as np
# Numba: compila a código máquina nativo
# prange: parallel range — equivalente a #pragma omp parallel for en C
from numba import njit, prange, float64, int32, int8

# ─────────────────────────────────────────────
# @njit(parallel=True) — compilado a código máquina
# prange — las hormigas corren en hilos nativos (no procesos)
#
# Por qué es más rápido que multiprocessing:
#   - Sin serialización de datos entre procesos
#   - Sin overhead de fork/join de procesos
#   - Los hilos comparten memoria igual que en tu proyecto C
#   - El loop interno también se compila a instrucciones SIMD
# ─────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _correr_colonia(pesos, valores, eta_base, feromonas,
                    capacidad, alfa, n,
                    num_hormigas, iteraciones, Rho, q):
    """
    Todo el algoritmo ACO compilado a código máquina.
    parallel=True hace que el prange de hormigas use hilos nativos.
    cache=True guarda la compilación en disco — la segunda ejecución
    arranca instantáneo.
    """
    mejor_val = 0
    mejor_sol = np.zeros(n, dtype=int8)

    # Buffer de soluciones de la iteración (hormigas x ítems)
    soluciones = np.zeros((num_hormigas, n), dtype=int8)

    for it in range(iteraciones):

        # ── prange: cada hormiga corre en un hilo nativo ──
        # Equivalente al pthread_create de tu proyecto C,
        # pero sin overhead de fork ni serialización
        for h in prange(num_hormigas):
            peso_actual = 0
            # Máscara booleana de candidatos para esta hormiga
            candidatos = np.ones(n, dtype=int8)

            while True:
                # Encontrar candidatos válidos
                hay_valido = False
                for i in range(n):
                    if candidatos[i] == 1 and pesos[i] <= capacidad - peso_actual:
                        hay_valido = True
                        break
                if not hay_valido:
                    break

                # Calcular atractivos solo para candidatos válidos
                suma = 0.0
                atractivos = np.zeros(n, dtype=float64)
                for i in range(n):
                    if candidatos[i] == 1 and pesos[i] <= capacidad - peso_actual:
                        tau = feromonas[i] ** alfa
                        atractivos[i] = tau * eta_base[i]
                        suma += atractivos[i]

                # Selección por ruleta
                if suma == 0.0:
                    # Fallback: elegir el primero válido
                    for i in range(n):
                        if candidatos[i] == 1 and pesos[i] <= capacidad - peso_actual:
                            seleccionado = i
                            break
                else:
                    r = np.random.random() * suma
                    acum = 0.0
                    seleccionado = 0
                    for i in range(n):
                        if candidatos[i] == 1 and pesos[i] <= capacidad - peso_actual:
                            acum += atractivos[i]
                            if acum >= r:
                                seleccionado = i
                                break

                soluciones[h, seleccionado] = 1
                peso_actual += pesos[seleccionado]
                candidatos[seleccionado] = 0

        # ── Evaporación vectorizada ──
        for i in range(n):
            feromonas[i] *= (1.0 - Rho)

        # ── Acciones demonio: encontrar mejor de la iteración ──
        for h in range(num_hormigas):
            val = 0
            for i in range(n):
                val += soluciones[h, i] * valores[i]
            if val > mejor_val:
                mejor_val = val
                for i in range(n):
                    mejor_sol[i] = soluciones[h, i]

        # ── Refuerzo elitista sobre mejor global ──
        refuerzo = q * mejor_val / 100.0
        for i in range(n):
            if mejor_sol[i] == 1:
                feromonas[i] += refuerzo

        # Reset buffer para siguiente iteración
        for h in range(num_hormigas):
            for i in range(n):
                soluciones[h, i] = 0

    return mejor_val, mejor_sol


class ACO_Mochila:
    def __init__(self, pesos, valores, capacidad, alfa=1.0, beta=2.0, Rho=0.1, q=10):
        self.capacidad = capacidad
        self.n         = len(pesos)
        self.alfa      = alfa
        self.beta      = beta
        self.Rho       = Rho
        self.q         = q

        self.pesos    = np.array(pesos,   dtype=np.int32)
        self.valores  = np.array(valores, dtype=np.int32)

        # eta precalculada una sola vez — igual que en la versión numpy
        self.eta_base = (self.valores / self.pesos) ** self.beta

        self.feromonas             = np.ones(self.n, dtype=np.float64)
        self.mejor_solucion_global = None
        self.mejor_valor_global    = 0

    def resolver(self, iteraciones=50, num_hormigas=10):
        print("Compilando con Numba (solo la primera vez)...")
        t_comp = time.time()

        # Numba compila en el primer llamado — los siguientes son instantáneos
        # cache=True guarda la compilación en disco (.numba_cache/)
        mejor_val, mejor_sol = _correr_colonia(
            self.pesos, self.valores, self.eta_base, self.feromonas,
            self.capacidad, self.alfa, self.n,
            num_hormigas, iteraciones, self.Rho, self.q
        )

        print(f"Compilación + ejecución: {time.time() - t_comp:.2f}s")
        self.mejor_valor_global    = int(mejor_val)
        self.mejor_solucion_global = mejor_sol.tolist()
        print("\nProceso completado.")
        return self.mejor_solucion_global, self.mejor_valor_global


def leer_instancia_txt(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        lineas = [l.strip() for l in f.readlines() if l.strip()]
    capacidad        = int(lineas[0])
    valor_optimo_ref = int(lineas[1])
    pesos            = [int(x) for x in lineas[2].replace(',', ' ').split()]
    valores          = [int(x) for x in lineas[3].replace(',', ' ').split()]
    return capacidad, pesos, valores, valor_optimo_ref


if __name__ == "__main__":
    archivo = "knapsack_casos/k1500.txt"

    try:
        cap_max, p, v, opt = leer_instancia_txt(archivo)

        inicio = time.time()

        colonia = ACO_Mochila(
            p, v, cap_max,
            alfa = 1, beta = 5, Rho = 0.5 ,q = 5,
        )
        sol_final, val_final = colonia.resolver(iteraciones = 100, num_hormigas=1500)

        fin = time.time()

        print(f"Resultado para {archivo}:")
        print(f"Valor obtenido:          {val_final}")
        print(f"Valor óptimo referencia: {opt}")
        print(f"Diferencia:              {opt - val_final}")
        print(f"Tiempo de ejecución:     {fin - inicio:.2f} segundos")

    except FileNotFoundError:
        print(f"No se encontró este archivo {archivo}")