import random
import json
from rf_model import predict_price_rf, train_optimized_rf_model
from sklearn.metrics import mean_absolute_percentage_error


class GAOptimizer:
    def __init__(self):
        self.initial_population = {"Individu": [], "Parameter": []}
        self.generations_data = {
            "Generasi": [],
            "Best Fitness": [],
            "MAPE": [],
            "Mutation": [],
        }
        self.evaluation_data = {}
        self.best_individuals = {}

    def save_log(self, generation, individual_index, log_type, log_content):
        if log_type == "initial":
            self.initial_population.setdefault("Individu", []).append(
                f"Individu {individual_index + 1}"
            )
            self.initial_population.setdefault("Parameter", []).append(log_content)

        elif log_type == "fitness":
            gen_data = self.evaluation_data.setdefault(
                generation, {"Individu": [], "MAPE (%)": [], "Fitness": []}
            )
            gen_data["Individu"].append(f"Individu {individual_index + 1}")
            gen_data["MAPE (%)"].append(log_content.get("MAPE"))
            gen_data["Fitness"].append(log_content.get("Fitness"))

        elif log_type in ["crossover", "mutation"]:
            gen_data = self.generations_data.setdefault(generation, {})
            gen_data.setdefault(log_type.capitalize(), []).append(log_content)

        elif log_type == "elitism":
            self.best_individuals[generation] = log_content

        elif log_type == "final":
            self.generations_data.setdefault("Generasi", []).append(generation)
            self.generations_data.setdefault("Best Fitness", []).append(
                log_content.get("Best Fitness")
            )
            self.generations_data.setdefault("MAPE", []).append(log_content.get("MAPE"))

    def evaluate_rf_model(self, params, data, price_target):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params

        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth > 0 else None
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)

        model = train_optimized_rf_model(data, price_target, params)

        predictions = []
        for i in range(len(data)):
            prediction = predict_price_rf(
                model,
                data.iloc[i]["bedroom"],
                data.iloc[i]["bathroom"],
                data.iloc[i]["LT"],
                data.iloc[i]["LB"],
                fallback_value=price_target.iloc[i],
            )
            predictions.append(prediction)

        mape = mean_absolute_percentage_error(price_target, predictions)
        return mape

    def fitness_function(self, params, data, price_target):
        return -self.evaluate_rf_model(params, data, price_target)

    def create_initial_population(self, size, object_bounds):
        population = []
        for _ in range(size):
            individual = tuple(
                random.uniform(lower_bound, upper_bound)
                for lower_bound, upper_bound in object_bounds
            )
            population.append(individual)
        return population

    def selection(self, population, fitnesses, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(
                list(zip(population, fitnesses)), tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        alpha = random.random()
        child1 = tuple(
            alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)
        )
        child2 = tuple(
            alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2)
        )
        return child1, child2

    def mutation(self, individual, mutation_rate, object_bounds):
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                lower_bound, upper_bound = object_bounds[i]
                mutation_amount = random.uniform(-0.1, 0.1) * (
                    upper_bound - lower_bound
                )
                individual[i] += mutation_amount
                individual[i] = max(min(individual[i], upper_bound), lower_bound)
        return tuple(individual)

    def genetic_algorithm(
        self,
        population_size,
        object_bounds,
        generations,
        mutation_rate,
        data,
        price_target,
    ):
        population = self.create_initial_population(population_size, object_bounds)

        for i, ind in enumerate(population):
            self.save_log(
                generation=0, individual_index=i, log_type="initial", log_content=ind
            )

        best_performers = []

        for generation in range(1, generations + 1):
            print(f"Generation {generation}/{generations}")

            fitnesses = [
                self.fitness_function(ind, data, price_target=price_target)
                for ind in population
            ]
            for i, (ind, fitness) in enumerate(zip(population, fitnesses)):
                log_content = {"MAPE": -fitness, "Fitness": fitness}
                self.save_log(generation, i, "fitness", log_content)

            best_index = fitnesses.index(max(fitnesses))
            best_individual = population[best_index]
            best_fitness = fitnesses[best_index]
            best_mape = -best_fitness
            best_performers.append((best_individual, best_fitness))

            self.generations_data["Generasi"].append(generation)
            self.generations_data["Best Fitness"].append(best_fitness)
            self.generations_data["MAPE"].append(best_mape)

            best_individual_data = {
                "Parameter": best_individual,
                "Fitness": best_fitness,
            }
            self.save_log(generation, best_index, "elitism", best_individual_data)

            selected_population = self.selection(population, fitnesses)

            next_population = []
            self.save_log(generation, -1, "crossover", "Crossover Phase")
            for i in range(0, len(selected_population) - 1, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                crossover_info = {
                    "Parent1": parent1,
                    "Parent2": parent2,
                    "Child1": child1,
                    "Child2": child2,
                }
                self.save_log(generation, i, "crossover", crossover_info)
                next_population.extend([child1, child2])

            if len(selected_population) % 2 != 0:
                next_population.append(selected_population[-1])
                self.save_log(
                    generation,
                    -1,
                    "crossover",
                    f"Odd individual carried to next generation: {selected_population[-1]}",
                )

            mutated_population = []
            self.save_log(generation, -1, "mutation", "Mutation Phase")
            for i, ind in enumerate(next_population):
                mutated = self.mutation(ind, mutation_rate, object_bounds)
                mutation_info = {"Original": ind, "Mutated": mutated}
                self.save_log(generation, i, "mutation", mutation_info)
                mutated_population.append(mutated)

            mutated_population[0] = best_individual
            self.save_log(generation, 0, "elitism", best_individual)

            population = mutated_population

        final_best_individual = max(best_performers, key=lambda x: x[1])[0]
        final_best_fitness = max(best_performers, key=lambda x: x[1])[1]
        final_best_mape = -final_best_fitness
        final_results = {
            "Best Individual": final_best_individual,
            "Fitness": final_best_fitness,
            "MAPE": final_best_mape,
        }
        self.save_log(generations, -1, "final", final_results)

        return final_best_individual, best_performers

    def save_logs_to_json(self, file_path):
        data = {
            "initial_population": self.initial_population,
            "evaluation_data": self.evaluation_data,
            "generations_data": self.generations_data,
            "best_individuals": self.best_individuals,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
