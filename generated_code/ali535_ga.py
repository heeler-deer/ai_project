def priority(current_city, distances, visited):
    """Calculate complex priorities for each city from the current city based on negative distance and other factors."""
    """Calculate refined priorities for cities based on multiple factors, including distances, visitation state, and nearby cities."""
    num_cities = len(distances)
    priorities = np.full(num_cities, np.inf)

    # Precompute total and immediate distances
    total_distances = np.sum(distances, axis=1)
    immediate_distances = distances[current_city, :]

    for city in range(num_cities):
        if not visited[city]:
            distance_to_city = immediate_distances[city]
            distance_priority = -distance_to_city

            dynamic_cost = 1 / (1 + total_distances[city]) if total_distances[city] > 0 else 0

            nearby_bonus = 0
            close_neighbors = 0
            distant_neighbors = 0

            for neighbor in range(num_cities):
                if not visited[neighbor]:
                    distance_to_neighbor = distances[city][neighbor]
                    if distance_to_neighbor < 3:
                        nearby_bonus += 10  # Significant bonus for very close cities
                        close_neighbors += 1
                    elif distance_to_neighbor < 5:
                        nearby_bonus += 4  # Moderate bonus for nearby cities
                    elif distance_to_neighbor < 10:
                        distant_neighbors += 1  # Count distant neighbors for strategic decisions

            # Average nearby bonus if close neighbors exist
            if close_neighbors > 0:
                nearby_bonus /= close_neighbors

            priorities[city] = distance_priority * dynamic_cost + nearby_bonus

            # Dynamic penalty for distances over thresholds
            if distance_to_city > 15:
                penalties = (distance_to_city - 15) / 2  # Reduce priority based on how far it is
                priorities[city] += penalties
            if distance_to_city > 25:
                penalties += (distance_to_city - 25) / 5  # Further penalty for very far cities
                priorities[city] += penalties

            # Heavy penalty for visited cities
            if visited[city]:
                priorities[city] += 100

            # Additional strategic adjustments based on nearby city counts
            if close_neighbors > 0:
                priorities[city] *= 0.85  # Strongly favor cities with more close neighbors
            if distant_neighbors > 2:
                priorities[city] += 5  # Encouragement to explore distant options if they are plentiful

            # Weight adjustments based on distance range
            if distance_to_city <= 5:
                priorities[city] *= 0.8  # Strong favor for very close cities
            elif 10 < distance_to_city <= 15:
                priorities[city] *= 1.1  # Slightly favor moderate distances

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
