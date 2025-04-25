# -513   57 p=0.5   eli51
def priority(current_city, distances, visited):
    """Calculate complex priorities for each city from the current city based on negative distance and other factors."""
    """Enhanced version of priority calculations with additional behavioral and efficiency improvements."""
    num_cities = len(distances)
    priorities = np.full(num_cities, np.inf)

    # Define weights for various factors
    weights = {
        'population_density': 0.4,
        'city_attractiveness': 0.4,
        'dynamic_cost': 0.2
    }

    # Precompute total distances and other factors for efficiency
    total_distance = np.sum(distances[current_city]) if np.sum(distances[current_city]) > 0 else 1
    attractiveness_base = np.random.random(num_cities)
    population_density_base = np.random.random(num_cities)

    dynamic_costs = np.zeros(num_cities)
    for city in range(num_cities):
        if not visited[city]:
            dynamic_costs[city] = 1 / (1 + np.sum(distances[city]))

    for city in range(num_cities):
        if visited[city]:
            continue  # Skip visited cities

        distance_priority = -distances[current_city][city]
        dynamic_cost = dynamic_costs[city]

        # Calculate the attractiveness score
        attr_factor = (population_density_base[city] * weights['population_density'] +
                       attractiveness_base[city] * weights['city_attractiveness'])
        attractiveness_score = (1 / (1 + attr_factor)) if attr_factor > 0 else 1

        # Combine all factors into the total priority
        combined_priority = (weights['dynamic_cost'] * dynamic_cost +
                             weights['population_density'] * population_density_base[city] +
                             weights['city_attractiveness'] * attractiveness_base[city])

        # Adjust for distance decay
        distance_decay = np.exp(-distances[current_city][city] / total_distance)
        adjusted_priority = (distance_priority + combined_priority) * distance_decay

        # Conditional logic to assign and adjust priorities
        if adjusted_priority < priorities[city]:
            priorities[city] = adjusted_priority
        elif adjusted_priority >= priorities[city]:
            if attractiveness_score < 0.4:  # Modify threshold for more sensitivity
                priorities[city] *= (1 + attractiveness_score)

            # Introduce a penalty if the distance is significantly high
            if distances[current_city][city] > (total_distance / 2):
                priorities[city] *= 0.9  # Apply a penalty of 10% for distant cities

    # Normalize priorities with consideration for potential zero entries
    valid_priorities = priorities[priorities < np.inf]
    max_priority = np.max(valid_priorities) if valid_priorities.size > 0 else 1
    priorities /= max_priority

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
filepath = ['../smaples_data/eil51.tsp']
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