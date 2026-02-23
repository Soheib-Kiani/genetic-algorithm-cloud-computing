import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import os

# ============================================================================
# TASK SCHEDULER CLASS
# ============================================================================

class TaskScheduler:
    def __init__(self, n_tasks: int, n_vms: int):
        self.n_tasks = n_tasks
        self.n_vms = n_vms      
        self.task_requirements = np.random.uniform(1000, 100000, n_tasks)     
        self.vm_capacities = np.random.uniform(2000, 16000, n_vms)
    
    def calculate_makespan(self, assignment: np.ndarray) -> float:
        vm_times = np.zeros(self.n_vms)       
        for task_id, vm_id in enumerate(assignment):
            execution_time = self.task_requirements[task_id] / self.vm_capacities[vm_id]
            vm_times[vm_id] += execution_time   
        return np.max(vm_times)
    
    def calculate_fitness(self, assignment: np.ndarray) -> float:
        """
        Higher fitness = better solution  => make span LOW -> fintness HIGH -> high chance to pick GA
        """
        makespan = self.calculate_makespan(assignment)
        return 1.0 / makespan
    # ?????????
        # finess = 1/(alpha * makespan + beta * energy) => Time + Energy consumption ; alpha(time) beta(energy)

# ============================================================================
#  GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm:
    def __init__(self, scheduler: TaskScheduler, population_size: int = 100,
                 generations: int = 200, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1):
        self.scheduler = scheduler
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.name = "Simple GA"
    
    def initialize_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size):
            individual = np.random.randint(0, self.scheduler.n_vms, 
                                          self.scheduler.n_tasks)
            population.append(individual)
        return population
    
    def tournament_selection(self, population: List[np.ndarray], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> np.ndarray:
        """
            tournament_fitness = [0.02, 0.009, 0.03]
            np.argmax(...) = 2
            winner_idx = tournament_indices[2] = 4
        """
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def single_point_crossover(self, parent1: np.ndarray, 
                               parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutate(self, individual: np.ndarray, mutation_rate: float = None) -> np.ndarray:
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.randint(0, self.scheduler.n_vms)
        return mutated
    
    def run(self) -> tuple:
        population = self.initialize_population()
        best_makespans = []
        avg_makespans = []
        
        for generation in range(self.generations):
            fitness_scores = [self.scheduler.calculate_fitness(ind) 
                            for ind in population]
            
            makespans = [self.scheduler.calculate_makespan(ind) 
                        for ind in population]
            best_makespans.append(min(makespans))
            avg_makespans.append(np.mean(makespans))
            
            new_population = []
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        final_fitness = [self.scheduler.calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx], best_makespans, avg_makespans


# ============================================================================
# IMPROVED GENETIC ALGORITHM
# ============================================================================

class ImprovedGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, scheduler: TaskScheduler, population_size: int = 100,
                 generations: int = 200, crossover_rate: float = 0.8,
                 initial_mutation_rate: float = 0.1, elitism_count: int = 2):
        super().__init__(scheduler, population_size, generations, 
                        crossover_rate, initial_mutation_rate)
        self.elitism_count = elitism_count
        self.name = "Improved GA"
    
    def weighted_round_robin_assignment(self) -> np.ndarray:
        weights = self.scheduler.vm_capacities / np.sum(self.scheduler.vm_capacities)
    
        tasks_per_vm = (weights * self.scheduler.n_tasks).astype(int)
        
        remaining_tasks = self.scheduler.n_tasks - np.sum(tasks_per_vm)
        if remaining_tasks > 0:
            sorted_vm_indices = np.argsort(self.scheduler.vm_capacities)[::-1]
            for i in range(remaining_tasks):
                tasks_per_vm[sorted_vm_indices[i % self.scheduler.n_vms]] += 1
        
        individual = np.zeros(self.scheduler.n_tasks, dtype=int)
        task_idx = 0
        for vm_id in range(self.scheduler.n_vms):
            for _ in range(tasks_per_vm[vm_id]):
                if task_idx < self.scheduler.n_tasks:
                    individual[task_idx] = vm_id
                    task_idx += 1
        
        np.random.shuffle(individual)
        return individual
    
    def initialize_population(self) -> List[np.ndarray]:
        population = []
        
        for _ in range(self.population_size):
            individual = self.weighted_round_robin_assignment()
            
            num_swaps = np.random.randint(
                int(0.05 * self.scheduler.n_tasks),
                int(0.10 * self.scheduler.n_tasks) + 1
            )
            
            for _ in range(num_swaps):
                task_idx = np.random.randint(0, self.scheduler.n_tasks)
                individual[task_idx] = np.random.randint(0, self.scheduler.n_vms)
            
            population.append(individual)
        
        return population
    
    def run(self) -> tuple:
        population = self.initialize_population()
        best_makespans = []
        avg_makespans = []
        
        for generation in range(self.generations):
            fitness_scores = [self.scheduler.calculate_fitness(ind) 
                            for ind in population]
            
            makespans = [self.scheduler.calculate_makespan(ind) 
                        for ind in population]
            best_makespans.append(min(makespans))
            avg_makespans.append(np.mean(makespans))
            
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            elites = [population[i].copy() for i in elite_indices]

            new_population = []
            
            while len(new_population) < self.population_size - self.elitism_count:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                current_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
                
                child1 = self.mutate(child1, current_mutation_rate)
                child2 = self.mutate(child2, current_mutation_rate)
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:self.population_size - self.elitism_count]
            new_population.extend(elites)
            
            population = new_population
        
        final_fitness = [self.scheduler.calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx], best_makespans, avg_makespans


# ============================================================================
# OPTIMIZED HYBRID GENETIC ALGORITHM
# ============================================================================

class OptimizedHybridGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, scheduler: TaskScheduler, population_size: int = 100,
                 generations: int = 200, crossover_rate: float = 0.8,
                 initial_mutation_rate: float = 0.1, elitism_count: int = 2,
                 local_search_rate: float = 0.1,
                 local_search_iterations: int = 3):
        super().__init__(scheduler, population_size, generations, 
                        crossover_rate, initial_mutation_rate)
        self.elitism_count = elitism_count
        self.local_search_rate = local_search_rate
        self.local_search_iterations = local_search_iterations
        self.name = "Optimized HGA"
    
    def weighted_round_robin_assignment(self) -> np.ndarray:
        weights = self.scheduler.vm_capacities / np.sum(self.scheduler.vm_capacities)
        
        tasks_per_vm = (weights * self.scheduler.n_tasks).astype(int)
        
        remaining_tasks = self.scheduler.n_tasks - np.sum(tasks_per_vm)
        if remaining_tasks > 0:
            sorted_vm_indices = np.argsort(self.scheduler.vm_capacities)[::-1]
            for i in range(remaining_tasks):
                tasks_per_vm[sorted_vm_indices[i % self.scheduler.n_vms]] += 1
        
        individual = np.zeros(self.scheduler.n_tasks, dtype=int)
        task_idx = 0
        for vm_id in range(self.scheduler.n_vms):
            for _ in range(tasks_per_vm[vm_id]):
                if task_idx < self.scheduler.n_tasks:
                    individual[task_idx] = vm_id
                    task_idx += 1
        
        np.random.shuffle(individual)
        return individual
    
    def initialize_population(self) -> List[np.ndarray]:
        population = []
        
        for _ in range(self.population_size):
            individual = self.weighted_round_robin_assignment()

            num_swaps = np.random.randint(
                int(0.05 * self.scheduler.n_tasks),
                int(0.10 * self.scheduler.n_tasks) + 1
            )
            
            for _ in range(num_swaps):
                task_idx = np.random.randint(0, self.scheduler.n_tasks)
                individual[task_idx] = np.random.randint(0, self.scheduler.n_vms)
            
            population.append(individual)
        
        return population
    
    def fast_local_search(self, individual: np.ndarray) -> np.ndarray:
        best_individual = individual.copy()
        best_makespan = self.scheduler.calculate_makespan(best_individual)
        
        sample_size = min(20, max(5, self.scheduler.n_tasks // 10))
        # 5 task minimum or 10% if task are more and 20 for huge tasks if we have
        
        for _ in range(self.local_search_iterations):
            improved = False
            
            sampled_tasks = np.random.choice(self.scheduler.n_tasks, 
                                            size=sample_size, 
                                            replace=False)
            
            for task in sampled_tasks:
                current_vm = best_individual[task]
                
                vm_candidates = np.random.choice(
                    [vm for vm in range(self.scheduler.n_vms) if vm != current_vm],
                    size=min(3, self.scheduler.n_vms - 1),
                    # Selected task will be tested in 3 VMs
                    replace=False
                )
                
                for new_vm in vm_candidates:
                    neighbor = best_individual.copy()
                    neighbor[task] = new_vm
                    
                    neighbor_makespan = self.scheduler.calculate_makespan(neighbor)
                    
                    if neighbor_makespan < best_makespan:
                        best_individual = neighbor
                        best_makespan = neighbor_makespan
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_individual
    
    def apply_local_search(self, population: List[np.ndarray], generation: int) -> List[np.ndarray]:
        if generation < 50:
            if generation % 10 != 0:
                return population
        else:
            if generation % 5 != 0:
                return population
            
        """
        Befor generation 50 => do thing after each 10 generation 
        After generation 50 => do thing after each  5 generation 
        """
        
        fitness_scores = [self.scheduler.calculate_fitness(ind) for ind in population]
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        num_to_search = int(self.population_size * self.local_search_rate)
        
        improved_population = population.copy()
        
        for i in range(num_to_search):
            idx = sorted_indices[i]
            improved_population[idx] = self.fast_local_search(population[idx])
        
        return improved_population
    
    def run(self) -> tuple:
        population = self.initialize_population()
        best_makespans = []
        avg_makespans = []
        
        for generation in range(self.generations):
            fitness_scores = [self.scheduler.calculate_fitness(ind) 
                            for ind in population]
            
            makespans = [self.scheduler.calculate_makespan(ind) 
                        for ind in population]
            best_makespans.append(min(makespans))
            avg_makespans.append(np.mean(makespans))
            
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            elites = [population[i].copy() for i in elite_indices]
            
            new_population = []
            
            while len(new_population) < self.population_size - self.elitism_count:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                current_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
                
                child1 = self.mutate(child1, current_mutation_rate)
                child2 = self.mutate(child2, current_mutation_rate)
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:self.population_size - self.elitism_count]
            
            new_population.extend(elites)
            
            population = self.apply_local_search(new_population, generation)
        
        final_fitness = [self.scheduler.calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx], best_makespans, avg_makespans


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_experiment_with_three_gas(n_tasks_list: List[int], n_vms_list: List[int], 
                                  num_runs: int = 5) -> pd.DataFrame:
    results = []
    
    for n_tasks in n_tasks_list:
        for n_vms in n_vms_list:
            print(f"\n{'='*70}")
            print(f"Configuration: Tasks={n_tasks}, VMs={n_vms}")
            print(f"{'='*70}")
            
            ga_makespans = []
            iga_makespans = []
            hga_makespans = []
            
            ga_times = []
            iga_times = []
            hga_times = []
            
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}...", end=' ')
                
                scheduler = TaskScheduler(n_tasks, n_vms)
                
                # 1. Simple Genetic Algorithm
                start_time = time.time()
                ga = GeneticAlgorithm(scheduler, population_size=100, 
                                     generations=200, crossover_rate=0.8, 
                                     mutation_rate=0.1)
                best_solution_ga, _, _ = ga.run()
                ga_time = time.time() - start_time
                makespan_ga = scheduler.calculate_makespan(best_solution_ga)
                ga_makespans.append(makespan_ga)
                ga_times.append(ga_time)
                
                # 2. Improved Genetic Algorithm
                start_time = time.time()
                iga = ImprovedGeneticAlgorithm(scheduler, population_size=100, 
                                              generations=200, crossover_rate=0.8,
                                              initial_mutation_rate=0.1, 
                                              elitism_count=2)
                best_solution_iga, _, _ = iga.run()
                iga_time = time.time() - start_time
                makespan_iga = scheduler.calculate_makespan(best_solution_iga)
                iga_makespans.append(makespan_iga)
                iga_times.append(iga_time)
                
                # 3. Optimized Hybrid Genetic Algorithm
                start_time = time.time()
                hga = OptimizedHybridGeneticAlgorithm(scheduler, population_size=100,
                                                     generations=200, crossover_rate=0.8,
                                                     initial_mutation_rate=0.1,
                                                     elitism_count=2,
                                                     local_search_rate=0.1,
                                                     local_search_iterations=3)
                best_solution_hga, _, _ = hga.run()
                hga_time = time.time() - start_time
                makespan_hga = scheduler.calculate_makespan(best_solution_hga)
                hga_makespans.append(makespan_hga)
                hga_times.append(hga_time)
                
                print(f"GA: {makespan_ga:.2f} ({ga_time:.1f}s) | IGA: {makespan_iga:.2f} ({iga_time:.1f}s) | HGA: {makespan_hga:.2f} ({hga_time:.1f}s)")
            
            # Calculate statistics
            results.append({
                'Tasks': n_tasks,
                'VMs': n_vms,
                'GA_Mean': np.mean(ga_makespans),
                'GA_Std': np.std(ga_makespans),
                'GA_Min': np.min(ga_makespans),
                'GA_Max': np.max(ga_makespans),
                'GA_Time': np.mean(ga_times),
                'IGA_Mean': np.mean(iga_makespans),
                'IGA_Std': np.std(iga_makespans),
                'IGA_Min': np.min(iga_makespans),
                'IGA_Max': np.max(iga_makespans),
                'IGA_Time': np.mean(iga_times),
                'HGA_Mean': np.mean(hga_makespans),
                'HGA_Std': np.std(hga_makespans),
                'HGA_Min': np.min(hga_makespans),
                'HGA_Max': np.max(hga_makespans),
                'HGA_Time': np.mean(hga_times),
                'IGA_vs_GA_Improvement_%': ((np.mean(ga_makespans) - np.mean(iga_makespans)) / 
                                            np.mean(ga_makespans) * 100),
                'HGA_vs_GA_Improvement_%': ((np.mean(ga_makespans) - np.mean(hga_makespans)) / 
                                            np.mean(ga_makespans) * 100),
                'HGA_vs_IGA_Improvement_%': ((np.mean(iga_makespans) - np.mean(hga_makespans)) / 
                                             np.mean(iga_makespans) * 100)
            })
            
            print(f"\n  Summary:")
            print(f"    GA:  {np.mean(ga_makespans):.2f} ± {np.std(ga_makespans):.2f} (Time: {np.mean(ga_times):.2f}s)")
            print(f"    IGA: {np.mean(iga_makespans):.2f} ± {np.std(iga_makespans):.2f} (Time: {np.mean(iga_times):.2f}s)")
            print(f"    HGA: {np.mean(hga_makespans):.2f} ± {np.std(hga_makespans):.2f} (Time: {np.mean(hga_times):.2f}s)")
    
    return pd.DataFrame(results)


def convergence_analysis_three_algorithms(n_tasks: int = 200, n_vms: int = 25):
    print(f"\n{'='*70}")
    print(f"Convergence Analysis: Tasks={n_tasks}, VMs={n_vms}")
    print(f"{'='*70}")
    
    scheduler = TaskScheduler(n_tasks, n_vms)
    
    print("\nRunning Simple GA...")
    ga = GeneticAlgorithm(scheduler, population_size=100, 
                         generations=100, crossover_rate=0.8, 
                         mutation_rate=0.1)
    _, ga_best, ga_avg = ga.run()
    
    print("Running Improved GA (with WRR)...")
    iga = ImprovedGeneticAlgorithm(scheduler, population_size=100, 
                                  generations=100, crossover_rate=0.8,
                                  initial_mutation_rate=0.1, 
                                  elitism_count=2)
    _, iga_best, iga_avg = iga.run()
    
    print("Running Optimized HGA (with WRR)...")
    hga = OptimizedHybridGeneticAlgorithm(scheduler, population_size=100,
                                         generations=100, crossover_rate=0.8,
                                         initial_mutation_rate=0.1,
                                         elitism_count=2,
                                         local_search_rate=0.1,
                                         local_search_iterations=3)
    _, hga_best, hga_avg = hga.run()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    generations = range(len(ga_best))
    
    ax1.plot(generations, ga_best, label='Simple GA', linewidth=2, alpha=0.8)
    ax1.plot(generations, iga_best, label='Improved GA (WRR)', linewidth=2, alpha=0.8)
    ax1.plot(generations, hga_best, label='Optimized HGA (WRR)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Makespan', fontsize=12)
    ax1.set_title('Best Makespan Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(generations, ga_avg, label='Simple GA', linewidth=2, alpha=0.8)
    ax2.plot(generations, iga_avg, label='Improved GA (WRR)', linewidth=2, alpha=0.8)
    ax2.plot(generations, hga_avg, label='Optimized HGA (WRR)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Average Makespan', fontsize=12)
    ax2.set_title('Average Makespan Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    
    plt.savefig('/mnt/user-data/outputs/convergence_comparison.png', 
                dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print("\n✓ Convergence plot saved")
    
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"Simple GA:")
    print(f"  Final Best Makespan:    {ga_best[-1]:.2f}")
    print(f"  Final Average Makespan: {ga_avg[-1]:.2f}")
    print(f"\nImproved GA (WRR):")
    print(f"  Final Best Makespan:    {iga_best[-1]:.2f}")
    print(f"  Final Average Makespan: {iga_avg[-1]:.2f}")
    print(f"  Improvement over GA:    {((ga_best[-1] - iga_best[-1]) / ga_best[-1] * 100):.2f}%")
    print(f"\nOptimized HGA (WRR):")
    print(f"  Final Best Makespan:    {hga_best[-1]:.2f}")
    print(f"  Final Average Makespan: {hga_avg[-1]:.2f}")
    print(f"  Improvement over GA:    {((ga_best[-1] - hga_best[-1]) / ga_best[-1] * 100):.2f}%")
    print(f"  Improvement over IGA:   {((iga_best[-1] - hga_best[-1]) / iga_best[-1] * 100):.2f}%")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CLOUD TASK SCHEDULING - THREE GA COMPARISON")
    print("Initialization: Weighted Round Robin (WRR)")
    print("="*70)
    
    # First: Convergence Analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    convergence_analysis_three_algorithms(n_tasks=200, n_vms=25)
    
    # Experiment 1: Varying number of tasks
    print("\n" + "="*70)
    print("EXPERIMENT 1: Varying Number of Tasks")
    print("="*70)
    results_tasks = run_experiment_with_three_gas(
        n_tasks_list=[100],
        n_vms_list=[15],
        num_runs=500
    )
    
    # Experiment 2: Varying number of VMs
    print("\n" + "="*70)
    print("EXPERIMENT 2: Varying Number of VMs")
    print("="*70)
    results_vms = run_experiment_with_three_gas(
        n_tasks_list=[100],
        n_vms_list=[15],
        num_runs=500
    )
    
    # Combine results
    all_results = pd.concat([results_tasks, results_vms], ignore_index=True)
    
    # Ensure output directory exists
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    
    # Save to Excel
    all_results.to_excel('/mnt/user-data/outputs/results_three_gas.xlsx', index=False)
    
    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print("✓ Excel file: results_three_gas.xlsx")
    print("✓ Convergence plot: convergence_comparison.png")
    print("\nAll experiments completed successfully!")
