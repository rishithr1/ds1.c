from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict
import random, heapq

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.weights = {}
        self.heuristic = {}
        self.and_nodes = set()
        self.or_nodes = set()

    def add_edge(self, u, v, weight=1):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.weights[(u, v)] = self.weights[(v, u)] = weight

    def set_heuristic(self, node, value):
        self.heuristic[node] = value

    def print_graph(self):
        print("Graph adjacency list:")
        for node, neighbors in self.graph.items():
            print(f"{node}: {', '.join(neighbors)}")

    # BFS
    def bfs(self, start, goal):
        queue, visited = deque([[start]]), {start}
        while queue:
            path = queue.popleft()
            if (node := path[-1]) == goal:
                return path
            for n in self.graph[node]:
                if n not in visited:
                    visited.add(n)
                    queue.append(path + [n])
        return []

    # DFS
    def dfs(self, start, goal):
        stack, visited = [(start, [start])], set()
        while stack:
            node, path = stack.pop()
            if node == goal:
                return path
            if node not in visited:
                visited.add(node)
                stack.extend((n, path + [n]) for n in reversed(self.graph[node]) if n not in visited)
        return []

    # British Museum Search
    def british_museum(self, start, goal, max_iters=1000):
        best_path = min((self.random_walk(start, goal) for _ in range(max_iters)), key=len, default=[])
        return best_path if best_path and best_path[-1] == goal else []

    def random_walk(self, current, goal):
        path, visited = [current], {current}
        while current != goal and (neighbors := [n for n in self.graph[current] if n not in visited]):
            current = random.choice(neighbors)
            path.append(current)
            visited.add(current)
        return path

    # Beam Search
    def beam_search(self, start, goal, beam_width=2):
        queue = [(start, [start])]
        while queue:
            queue = sorted([(n, path + [n]) for _, path in queue for n in self.graph[path[-1]] if n not in path],
                           key=lambda x: self.heuristic.get(x[0], float('inf')))[:beam_width]
            if any(node == goal for node, path in queue):
                return next(path for node, path in queue if node == goal)
        return []

    # Hill Climbing
    def hill_climbing(self, start, goal):
        current, path = start, [start]
        while current != goal:
            neighbors = [n for n in self.graph[current] if n not in path]
            if not neighbors:
                break
            next_node = min(neighbors, key=lambda x: self.heuristic.get(x, float('inf')))
            if self.heuristic.get(next_node, float('inf')) >= self.heuristic.get(current, float('inf')):
                break
            current, path = next_node, path + [next_node]
        return path if current == goal else []

    # Oracle Search (basic)
    def oracle_search(self, start, goal):
        return self.branch_and_bound(start, goal)

    # Oracle with cost and heuristics
    def oracle_with_cost_heuristic(self, start, goal):
        return self.a_star(start, goal)

    # Branch and Bound
    def branch_and_bound(self, start, goal):
        queue = [(0, start, [start])]
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            for n in self.graph[node]:
                if n not in path:
                    heapq.heappush(queue, (cost + self.weights.get((node, n), 1), n, path + [n]))
        return []

    # Branch and Bound with extended list
    def branch_and_bound_extended(self, start, goal):
        queue, visited = [(0, start, [start])], set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            visited.add(node)
            for n in self.graph[node]:
                if n not in visited:
                    heapq.heappush(queue, (cost + self.weights.get((node, n), 1), n, path + [n]))
        return []

    # A* Search
    def a_star(self, start, goal):
        queue, visited = [(0, start, [start])], set()
        while queue:
            _, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            visited.add(node)
            for n in self.graph[node]:
                if n not in visited:
                    heapq.heappush(queue, (len(path) + self.heuristic.get(n, 0), n, path + [n]))
        return []

    def ao_star(self, start: str, goal: str) -> List[str]:
        def calculate_cost(node: str, visited: Set[str]) -> Tuple[float, List[str]]:
            if node == goal:
                return 0, [node]
            if node in visited:
                return float('inf'), []
            
            visited.add(node)
            
            if node in self.and_nodes:
                total_cost = 0
                total_path = [node]
                for neighbor in self.graph[node]:
                    cost, path = calculate_cost(neighbor, visited.copy())
                    total_cost += self.weights.get((node, neighbor), 1) + cost
                    total_path.extend(path)
                return total_cost, total_path
            else:
                best_cost, best_path = float('inf'), []
                for neighbor in self.graph[node]:
                    cost, path = calculate_cost(neighbor, visited.copy())
                    total_cost = self.weights.get((node, neighbor), 1) + cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [node] + path

                return best_cost, best_path

        _, path = calculate_cost(start, set())
        return path if path and path[-1] == goal else []

    # Best-First Search
    def best_first_search(self, start, goal):
        queue, visited = [(0, start, [start])], set()
        while queue:
            _, node, path = heapq.heappop(queue)
            if node == goal:
                return path
            visited.add(node)
            for n in self.graph[node]:
                if n not in visited:
                    heapq.heappush(queue, (self.heuristic.get(n, 0), n, path + [n]))
        return []

def main():
    g = Graph()
    g.add_edge('S', 'A', 3)
    g.add_edge('S', 'B', 2)
    g.add_edge('A', 'B', 1)
    g.add_edge('A', 'D', 4)
    g.add_edge('B', 'C', 2)
    g.add_edge('C', 'E', 3)
    g.add_edge('D', 'G', 5)
    for node, value in {'S': 11, 'A': 7, 'B': 6, 'C': 7, 'D': 5, 'E': 6, 'G': 0}.items():
        g.set_heuristic(node, value)
        
    g.print_graph()

    start, goal = 'S', 'G'
    algorithms = {
        'BFS': g.bfs, 'DFS': g.dfs, 'British Museum': g.british_museum,
        'Beam Search': g.beam_search, 'Hill Climbing': g.hill_climbing,
        'Oracle Search': g.oracle_search, 'Oracle with Cost & Heuristics': g.oracle_with_cost_heuristic,
        'Branch and Bound': g.branch_and_bound, 'Branch and Bound with Extended List': g.branch_and_bound_extended,
        'A*': g.a_star, 'AO*': g.ao_star, 'Best-First Search': g.best_first_search
    }

    for name, algo in algorithms.items():
        result = algo(start, goal)
        print(f"{name}: {result}")

if __name__ == "__main__":
    main()


# from collections import deque, defaultdict
# import heapq
# import random
# from typing import Dict, List, Set, Tuple, Optional

# class Graph:
#     def __init__(self):
#         self.graph = defaultdict(list)
#         self.heuristic = {}  # -> A* and  informed search algorithms
#         self.weights = {}    # -> weighted edges
#         self.and_nodes = set()  # -> For AO* algorithm
#         self.or_nodes = set() # -> For AO* algorithm

#     def add_edge(self, u: str, v: str, weight: int = 1):
#         self.graph[u].append(v)
#         self.graph[v].append(u)  # For undirected graph
#         self.weights[(u, v)] = weight
#         self.weights[(v, u)] = weight

#     def set_heuristic(self, node: str, value: float):
#         self.heuristic[node] = value

#     def print_graph(self):
#         print("Graph adjacency list:")
#         for node, neighbors in self.graph.items():
#             print(f"{node}: {', '.join(neighbors)}")

#     def bfs(self, start: str, goal: str) -> List[str]:
#         queue = deque([[start]])
#         visited = set([start])
        
#         while queue:
#             path = queue.popleft()
#             node = path[-1]
#             print(f"Processing Node: {node}")
#             print(f"Current Path: {path}")
            
#             if node == goal:
#                 return path
                
#             for neighbor in self.graph[node]:
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     new_path = list(path)
#                     new_path.append(neighbor)
#                     queue.append(new_path)
#                     print(f"Pushing Path: {new_path}")
#         return []

#     def british_museum_search(self, start: str, goal: str, max_iterations: int = 1000) -> List[str]:
#         """British Museum Search (Random Walk)"""
#         best_path = None
#         best_length = float('inf')
        
#         for _ in range(max_iterations):
#             current = start
#             path = [current]
#             visited = {start}
            
#             while current != goal and len(self.graph[current]) > 0:
#                 neighbors = [n for n in self.graph[current] if n not in visited]
#                 if not neighbors:
#                     break
                    
#                 current = random.choice(neighbors)
#                 path.append(current)
#                 visited.add(current)
                
#             if current == goal and len(path) < best_length:
#                 best_path = path
#                 best_length = len(path)
                
#         return best_path if best_path else []

#     def dfs(self, start: str, goal: str) -> List[str]:
#         """Depth-First Search"""
#         stack = [(start, [start])]
#         print(stack)
#         visited = set() ##Prevents Cycle
        
#         while stack: ##Continues as long as nodes are there to explore
#             node, path = stack.pop()
#             print(f"Popped: {node}, Current Path: {path}")
#             if node not in visited:
#                 if node == goal:
#                     return path
                    
#                 visited.add(node)
#                 print(f"Visited: {node}")
#                 for neighbor in reversed(self.graph[node]): ##to control traversal order
#                     if neighbor not in visited:
#                         stack.append((neighbor, path + [neighbor]))
#                         print(f"Added to stack: {neighbor}, Path will be: {path + [neighbor]}")
#         return []

#     def hill_climbing(self, start: str, goal: str) -> List[str]:
#         """Hill Climbing Search"""
#         current = start
#         path = [current]
#         print(f"Starting search from: {current}")
        
#         while current != goal:
#             neighbors = self.graph[current]
#             print(f"Current node: {current}")
#             print(f"Neighbors: {neighbors}")
#             if not neighbors:
#                 break
#             next_node = min(neighbors, key=lambda x: self.heuristic.get(x, float('inf')))
#             print(f"Selected next node: {next_node} with heuristic value: {self.heuristic.get(next_node, float('inf'))}")
            
#             if self.heuristic.get(next_node, float('inf')) >= self.heuristic.get(current, float('inf')):
#                 print("Reached a local minimum, stopping search.")
#                 break
                
#             current = next_node
#             path.append(current)
            
#         return path if path[-1] == goal else []

#     def branch_and_bound(self, start: str, goal: str) -> List[str]:
#         """Branch and Bound Search"""
#         queue = [(0, start, [start])]  # (cost, node, path)
#         visited = set()
        
#         while queue:
#             cost, node, path = heapq.heappop(queue) ##node with the lowest cost is extracted from the queue
#             print(f"Total Cost: {cost}, Current Node: {node}, Current Path: {path}")
            
#             if node == goal:
#                 return path
                
#             if node not in visited:
#                 visited.add(node)
#                 for neighbor in self.graph[node]:
#                     if neighbor not in visited:
#                         new_cost = cost + self.weights.get((node, neighbor), 1) ##each unvisited neighbor, the new cost to reach that neighbor is calculated.
#                         heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))
#         return []

#     def branch_and_bound_with_heuristic(self, start: str, goal: str) -> List[str]:
#         """Branch and Bound with Heuristic Estimation"""
#         queue = [(self.heuristic.get(start, 0), 0, start, [start])]  # (estimate, cost, node, path)
#         visited = set()
        
#         while queue:
#             _, cost, node, path = heapq.heappop(queue)
            
#             if node == goal:
#                 return path
                
#             if node not in visited:
#                 visited.add(node)
#                 for neighbor in self.graph[node]:
#                     if neighbor not in visited:
#                         new_cost = cost + self.weights.get((node, neighbor), 1)
#                         estimate = new_cost + self.heuristic.get(neighbor, 0)
#                         #print(f"  Processing Neighbor: '{neighbor}'")
#                         #print(f"  New Cost to '{neighbor}': {new_cost}, Heuristic: {self.heuristic.get(neighbor, 0)}, Total Estimate: {estimate}")
#                         heapq.heappush(queue, (estimate, new_cost, neighbor, path + [neighbor]))
#         return []
    
#     def ao_star(self, start: str, goal: str) -> List[str]:
#         def calculate_cost(node: str, visited: Set[str]) -> Tuple[float, List[str]]:
#             if node == goal:
#                 return 0, [node]
#             if node in visited:
#                 return float('inf'), []

#             visited.add(node)
#             if node in self.and_nodes:
#                 ##All children must be visited (AND node)
#                 total_cost = 0
#                 total_path = [node]
#                 for neighbor in self.graph[node]:
#                     cost, path = calculate_cost(neighbor, visited.copy())
#                     total_cost += cost + self.weights.get((node, neighbor), 1)
#                     total_path.extend(path)
#                 return total_cost, total_path
#             else:
#                 ##Choose the best child (OR node)
#                 min_cost = float('inf')
#                 best_path = []
#                 for neighbor in self.graph[node]:
#                     cost, path = calculate_cost(neighbor, visited.copy())
#                     total_cost = cost + self.weights.get((node, neighbor), 1)

#                     if total_cost < min_cost:
#                         min_cost = total_cost
#                         best_path = [node] + path

#                 return min_cost, best_path
        
#         _, path = calculate_cost(start, set())
#         return path if path else []

#     #strat -> A goal/destination -> G
#     def oracle(self, start: str, goal: str) -> List[str]:
#         """Oracle Search - Assumes perfect knowledge of the shortest path
#         Uses Dijkstra's algorithm since we have perfect knowledge"""
#         distances = {node: float('inf') for node in self.graph}
#         distances[start] = 0
#         predecessors = {node: None for node in self.graph}
#         pq = [(0, start)]
#         visited = set()

#         while pq:
#             current_distance, current_node = heapq.heappop(pq)

#             if current_node in visited:
#                 continue

#             visited.add(current_node)

#             if current_node == goal:
#                 break

#             for neighbor in self.graph[current_node]:
#                 distance = current_distance + self.weights.get((current_node, neighbor), 1)
#                 if distance < distances[neighbor]:
#                     distances[neighbor] = distance
#                     predecessors[neighbor] = current_node
#                     heapq.heappush(pq, (distance, neighbor))

#         ##Reconstruct path
#         path = []
#         current = goal
#         while current is not None:
#             path.append(current)
#             current = predecessors[current]
#         return list(reversed(path))


# def main():
    
#     g = Graph()
    
#     ##add edges to tthe graph
#     g.add_edge('S', 'A', 3)    # Cost of 3 from S to A
#     g.add_edge('S', 'B', 2)    # Cost of 2 from S to B
#     g.add_edge('A', 'B', 1)    # Cost of 1 from A to B
#     g.add_edge('A', 'D', 4)    # Cost of 4 from A to D
#     g.add_edge('B', 'C', 2)    # Cost of 2 from B to C
#     g.add_edge('C', 'E', 3)    # Cost of 3 from C to E
#     g.add_edge('D', 'G', 5)    # Cost of 5 from D to G
#     # Setting heuristic values for nodes
#     for node, value in {'S':11,'A': 7, 'B': 6, 'C': 7, 'D': 5, 'E': 6, 'G': 0}.items():
#         g.set_heuristic(node, value)
    
#     g.print_graph()
    
#     start, goal = 'S', 'G'
#     algorithms = [
#         ('BFS', g.bfs),
#         ('British Museum', g.british_museum_search),
#         ('DFS', g.dfs),
#         ('Hill Climbing', g.hill_climbing),
#         ('Branch & Bound', g.branch_and_bound),
#         #('Oracle', g.oracle_search),
#         ('Branch & Bound with Heuristic', g.branch_and_bound_with_heuristic),
#         #('A*', g.a_star)
#         ('AO*', g.ao_star),
#         ('Oracle', g.oracle)
#     ]
    
    
#     for name, algo in algorithms:
#         path = algo(start, goal)
#         print(f"{name}: {' -> '.join(path)}")

# if __name__ == "__main__":
#     main()




# class Node:
#     def __init__(self, value=None):
#         self.value = value
#         self.children = []
#         self.alpha = float('-inf')
#         self.beta = float('inf')
        
#     def add_child(self, child):
#         self.children.append(child)
        
#     def is_terminal(self):
#         return len(self.children) == 0

# def create_game_tree():
#     def input_node(level=0):
#         value = int(input(f"Enter value for {'Root' if level == 0 else 'Node'} at level {level}: "))
#         node = Node(value)
        
#         num_children = int(input(f"Enter the number of children for node with value {value}: "))
#         for i in range(num_children):
#             print(f"\nEntering child {i+1} of node {value}")
#             child_node = input_node(level + 1)
#             node.add_child(child_node)
        
#         return node
    
#     print("Create Game Tree:")
#     return input_node()

# def print_tree(node, level=0, prefix="Root: "):
#     """Print the tree structure for visualization"""
#     print("  " * level + prefix + str(node.value))
#     for i, child in enumerate(node.children):
#         print_tree(child, level + 1, f"Child {i+1}: ")

# def minimax(node, depth, maximizing_player, moves=None):
#     if moves is None:
#         moves = []
#     moves.append((node.value, depth, "MAX" if maximizing_player else "MIN"))
    
#     if depth == 0 or node.is_terminal():
#         return node.value, moves
    
#     if maximizing_player:
#         max_eval = float('-inf')
#         for child in node.children:
#             eval, child_moves = minimax(child, depth - 1, False, moves)
#             max_eval = max(max_eval, eval)
#         return max_eval, moves
#     else:
#         min_eval = float('inf')
#         for child in node.children:
#             eval, child_moves = minimax(child, depth - 1, True, moves)
#             min_eval = min(min_eval, eval)
#         return min_eval, moves

# def alpha_beta(node, depth, alpha, beta, maximizing_player, moves=None):
#     if moves is None:
#         moves = []
#     moves.append((node.value, depth, "MAX" if maximizing_player else "MIN", alpha, beta))
    
#     if depth == 0 or node.is_terminal():
#         return node.value, moves
    
#     if maximizing_player:
#         max_eval = float('-inf')
#         for child in node.children:
#             eval, child_moves = alpha_beta(child, depth - 1, alpha, beta, False, moves)
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 moves.append(("Pruned", depth, "MAX", alpha, beta))
#                 break
#         return max_eval, moves
#     else:
#         min_eval = float('inf')
#         for child in node.children:
#             eval, child_moves = alpha_beta(child, depth - 1, alpha, beta, True, moves)
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 moves.append(("Pruned", depth, "MIN", alpha, beta))
#                 break
#         return min_eval, moves

# def visualize_moves(moves, algorithm_name):
#     print(f"\n{algorithm_name} Algorithm Traversal:")
#     print("=" * 50)
#     print("Format: (Node Value, Depth, Player Type, Alpha, Beta)")
#     print("-" * 50)
    
#     for move in moves:
#         if len(move) == 3:
#             value, depth, player = move
#             print(f"Depth {depth}: {player} evaluates node {value}")
#         else:
#             value, depth, player, alpha, beta = move
#             if value == "Pruned":
#                 print(f"Depth {depth}: {player} pruned branch (α={alpha}, β={beta})")
#             else:
#                 print(f"Depth {depth}: {player} evaluates node {value} (α={alpha}, β={beta})")

# if __name__ == "__main__":
#     root = create_game_tree()
#     print("Game Tree Structure:")
#     print_tree(root)
    
#     depth = int(input("Enter the maximum depth for the algorithms: "))
    
#     # Minimax Run
#     print("\nRunning Minimax...")
#     minimax_value, minimax_moves = minimax(root, depth, True)
#     visualize_moves(minimax_moves, "Minimax")
#     print(f"Minimax Final Value: {minimax_value}")
    
#     # Alpha-Beta Pruning Run
#     print("\nRunning Alpha-Beta Pruning...")
#     alpha_beta_value, alpha_beta_moves = alpha_beta(root, depth, float('-inf'), float('inf'), True)
#     visualize_moves(alpha_beta_moves, "Alpha-Beta")
#     print(f"Alpha-Beta Final Value: {alpha_beta_value}")


