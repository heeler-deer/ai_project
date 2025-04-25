def priority(current_city, distances, visited):
    """Calculate complex priorities for each city from the current city based on negative distance and other factors."""
    """Advanced version of the priority calculation with more complex decision-making, heuristics, and optimizations."""
    num_cities = len(distances)
    priorities = np.full(num_cities, np.inf)
    total_distance = np.sum(distances[current_city])

    # Precompute common metrics
    city_sums = np.sum(distances, axis=1)
    nearby_cities = np.where(distances[current_city] < 50)[0]
    far_cities = np.where(distances[current_city] > 500)[0]
    special_bonus_cities = [city for city in range(num_cities) if city % 2 == 0 or city % 3 == 0]

    # Precompute the inverse of distances for faster calculations later
    inv_distances = 1 / (1 + distances[current_city])

    # Define a custom factor to adjust priorities based on the city type
    def calculate_feature_bonus(city):
        bonus = 0
        if city in special_bonus_cities:
            if city % 2 == 0:
                bonus += 0.7
            if city % 3 == 0:
                bonus += 0.4
        return bonus

    # Threshold-based adjustments and penalties for proximity
    def apply_proximity_penalties(city, proximity_range):
        proximity_penalty = 0
        if distances[current_city][city] < proximity_range:
            proximity_penalty += 0.1  # Small penalty for close cities
        return proximity_penalty

    # More dynamic cost calculation, factoring in neighboring cities and potential clusters
    def dynamic_cost(city, city_sums):
        base_cost = 1 / (1 + city_sums[city]) + np.sum(distances[city] < 50) * 0.2
        return base_cost

    # Time decay effect
    def time_decay(city):
        if distances[current_city][city] > 50:
            return np.exp(-0.1 * distances[current_city][city])
        return 1

    for city in range(num_cities):
        if not visited[city]:
            distance_priority = -distances[current_city][city]
            feature_bonus = calculate_feature_bonus(city)
            dynamic_cost_value = dynamic_cost(city, city_sums)
            decay = time_decay(city)

            proximity_penalty = apply_proximity_penalties(city, 10)
            cluster_penalty = 0
            for other_city in range(num_cities):
                if not visited[other_city] and other_city != city and abs(distances[current_city][city] - distances[current_city][other_city]) < 50:
                    cluster_penalty += 0.1

            # Applying the calculated factors
            priority = (distance_priority * dynamic_cost_value) + feature_bonus + proximity_penalty + cluster_penalty
            priority *= decay

            # Adjust priority based on total distance thresholds
            if total_distance < 100:
                priorities[city] = priority
            elif 100 <= total_distance < 300:
                priorities[city] = priority * 1.3
            elif total_distance >= 300:
                priorities[city] = priority * 0.8

            # More dynamic adjustments based on specific conditions
            if city in nearby_cities:
                priorities[city] += 0.2
            if city in far_cities:
                priorities[city] += 0.3

            # Weighted population-based adjustment
            population = np.random.randint(100000, 1000000)
            population_weight = 1 / (1 + population / 100000)
            priorities[city] *= population_weight

            # Special adjustment if total distance is exceptionally low or high
            if total_distance < 50:
                priorities[city] *= 0.85
            elif total_distance > 500:
                priorities[city] *= 1.25

    return priorities



import matplotlib.pyplot as plt
import sys, os

dataset = {}
def prepare_dataset(filename):
  # print(type(filename))
  with open(filename, "r") as f:
      lines = f.readlines()

  # Parsing node coordinate data
  node_coords = {}
  found_node_section = False
  for line in lines:
      if found_node_section:
          if line.strip() == "EOF":
              break
          node_id, x, y = map(float, line.strip().split())
          node_coords[node_id] = (x, y)
      elif line.startswith("NODE_COORD_SECTION"):
          found_node_section = True
  dataset[filename] = node_coords
  # print(node_coords)

# filepath = ['ai_project/smaples_data/ali535.tsp','/content/ai_project/smaples_data/eil51.tsp','/content/ai_project/smaples_data/lin318.tsp']
filepath = ['/content/ai_project/sample_data/gr137.tsp']
for i in filepath:
  prepare_dataset(i)
  
  
  import numpy as np
def coordinates_to_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    city_ids = sorted(coordinates.keys())

    # Calculate the Euclidean distance between each pair of cities
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            coord1 = coordinates[city_ids[i]]
            coord2 = coordinates[city_ids[j]]
            distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            distance_matrix[i][j] = distance_matrix[j][i] = distance

    return distance_matrix
# print(dataset)

input = {}
for i in dataset.keys():
    distance_matrix = coordinates_to_distance_matrix(dataset[i])
    input[i] = distance_matrix
print(input.keys())



# compare function

import numpy as np

def heuristic_tsp_solver(distances, priority_func):
    """Solve the TSP by selecting the next city based on a complex priority function."""
    num_cities = len(distances)
    visited = [False] * num_cities
    current_city = 0
    visited[current_city] = True
    tour = [current_city]
    total_distance = 0

    while len(tour) < num_cities:
        # Get complex priorities for the next city based on the current city
        priorities = priority_func(current_city, distances, visited)
        # Mask priorities for visited cities to ensure they are not selected
        masked_priorities = np.where(visited, np.inf, priorities)
        # Select the next city with the highest priority (lowest cost)
        next_city = np.argmin(masked_priorities)
        visited[next_city] = True
        tour.append(next_city)
        total_distance += distances[current_city][next_city]
        current_city = next_city

    # Close the loop by returning to the starting city
    total_distance += distances[current_city][tour[0]]
    tour.append(tour[0])  # Optionally return to the starting city for visualization
    return tour, total_distance


def evaluate(distance_matrix):
    """Evaluate heuristic function on a provided distance matrix for the TSP."""
    tour, total_distance = heuristic_tsp_solver(distance_matrix, priority)
    # return {'tour': tour, 'total_distance': total_distance}
    print(-total_distance)
    return -total_distance


evaluate(distance_matrix)
