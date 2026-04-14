import numpy as np
import time as tm
from numba import njit, prange

def read_knapsack(file_name):
    """formato"""
    with open(file_name, 'r') as f:
        objects = [ l.strip() for l in f.readlines() if l.strip() ];

    capacity = np.int64 (objects[ 0 ] );
    reference_value = np.int64( objects[ 1 ] );
    weight = [ np.int64( x ) for x in objects [ 2].replace( ',', ' ' ).split() ];
    values = [ np.int64( x ) for x in objects[ 3 ].replace( ',', ' ' ).split() ];

    return capacity, reference_value, weight, values;

@njit
def ant_decision_rule( mask : np.array, pheromones : np.array, eta_base : np.float64, alfa : np.float64):
        
        tau = pheromones[ mask ] ** alfa;
        eta = eta_base[ mask ];          # ya precalculada
        attractiveness = tau * eta;

        add_atractiveness = attractiveness.sum();
        valid_candidates = np.where( mask )[ 0 ];

        if add_atractiveness == 0.0:
            random_idx = np.random.randint( 0, len( valid_candidates ) )
            return valid_candidates[ random_idx ];

        probs = attractiveness / add_atractiveness

        r = np.random.random()

        cumulative = 0.0

        for i in range( len( probs ) ):

            cumulative += probs[ i ]
            
            if r <= cumulative:
                
                return valid_candidates[ i ]
                
        return valid_candidates[ -1 ] # Respaldo por errores mínimos de redondeo

@njit
def new_ant_activity( max_capacity : np.int64, weight_array : np.array, pheromones : np.array, eta_base : np.float64, alfa : np.float64 ):
        
        solution = np.zeros( weight_array.size, dtype = np.int64 );
        current_weight = 0;
        
        candidates = np.ones( weight_array.size, dtype = np.bool );

        while True:
            
            remaining_weight = max_capacity - current_weight;
            mask = candidates & ( weight_array <= remaining_weight );

            if not mask.any():
                break;

            choice = ant_decision_rule( mask, pheromones, eta_base, alfa );

            solution[ choice ]   = 1
            current_weight += weight_array[ choice ]
            candidates[ choice ] = False

        return solution

@njit(parallel=True)
def create_ant( max_capacity : np.int64, weight_array : np.array, values_array : np.array, pheromones : np.array, eta_base : np.float64, alfa : np.float64 ):
        
        solutions = np.zeros((weight_array.size, weight_array.size), dtype=np.int64);

        for i in prange(weight_array.size):
        
            solutions[i] = new_ant_activity( max_capacity, weight_array, pheromones, eta_base, alfa );
        
        
        return solutions;


def update_pheromones( pheromones : np.array, rho : np.float64  ):

    copy_pheromones = pheromones.copy();

    copy_pheromones *= (1 - rho);

    return copy_pheromones;

def execute_daemon_actions( solutions : list, values : np.array, best_value_global : np.int64, best_solution_global : np.array, q : np.float64, pheromones : np.array ):
        # Calcular todos los valores de la iteración de una sola vez
        # Antes: sum(sol[i] * self.valores[i] for i in range(self.n)) por cada sol
        solutions_copy = np.array( solutions, dtype=np.int64);
        iteration_values = solutions_copy.dot(values);  # producto matricial vectorizado

        best_idx = int( np.argmax( iteration_values ) );
        best_value = int( iteration_values[ best_idx ] );
        current_best_value = best_value_global;
        current_best_solution = best_solution_global.copy();
        current_pheromones = pheromones.copy();

        if best_value > best_value_global:
            current_best_value = best_value;
            current_best_solution = solutions[best_idx];

        # Refuerzo vectorizado — antes era un for i in range(self.n)
        if current_best_solution is not None:
            best_np = np.array( current_best_solution, dtype = np.float64 );
            current_pheromones += best_np * ( q * current_best_solution / 100);
        
        return current_pheromones, current_best_value, current_best_solution;

def main_aco( alfa = 1, beta = 2, rho = 0.5 ,q = 5, iterations = 50 ):

    max_capacity = np.int64;
    reference_value = np.int64;

    max_capacity, reference_value, weight_list, values_list = read_knapsack("knapsack_casos/k1500.txt");

    weight_array = np.array( weight_list, dtype = np.int64 );
    values_array = np.array( values_list, dtype = np.int64 );

    best_solution = np.zeros( weight_array.size );
    value_best_solution = np.int64( 0 );

    eta_base = (values_array / weight_array) ** beta;

    pheromones = np.ones( weight_array.size, dtype = np.float64 );

    solutions = [];

    times_ant_create = [];

    times_update_pheromones = [];

    times_daemons_actions = [];

    for iteraction in range( iterations ):

        print( f"\rIteración #{iteraction + 1} de {iterations}", end="", flush=True )

        create_ant_init_time = tm.time();

        solutions = create_ant( max_capacity, weight_array, values_array, pheromones, eta_base, alfa );

        create_ant_final_time = tm.time();

        total_time_create_ant = create_ant_final_time - create_ant_init_time

        times_ant_create.append( total_time_create_ant );

        update_pheromones_init_time = tm.time();
        
        pheromones = update_pheromones( pheromones, rho);

        update_pheromones_final_time = tm.time();

        times_update_pheromones.append( update_pheromones_final_time - update_pheromones_init_time );

        daemons_actions_init_time = tm.time();
        
        pheromones, value_best_solution, best_solution = execute_daemon_actions( solutions, values_array, value_best_solution, best_solution, q, pheromones );

        daemons_actions_final_time = tm.time();
    
        times_daemons_actions.append( daemons_actions_final_time - daemons_actions_init_time );

    array_times_create_ant = np.array( times_ant_create, dtype = np.float64 );
    array_times_update_pheromones = np.array( times_update_pheromones, dtype = np.float64 );
    array_times_daemons_actions = np.array( times_daemons_actions, dtype = np.float64 );
    
    return best_solution, value_best_solution, reference_value, array_times_create_ant.mean(), array_times_update_pheromones.mean(), array_times_daemons_actions.mean();


    
init = tm.time()

best_solution, value_best, reference_value, ant_mean, pheromones_mean, daemons_mean = main_aco( alfa = 1, beta = 5, rho = 0.5 ,q = 5, iterations = 100 );

final = tm.time()

total_time = final - init

print( "Este es el caso de ", best_solution.size, " objetos" )
print( "Mejor solucion encontrada: ", best_solution );
print( "Valor asociado: ", value_best );
print( "Valor de referencia", reference_value );
print( "tiempo invertido: ", round(total_time, 2), " segundos" );
print( "Promedio de creacion de hormigas: ", ant_mean );
print( "Promedio de actualizacion de feromonas: ", pheromones_mean );
print( "Promedio de las acciones del demonio: ", daemons_mean );

