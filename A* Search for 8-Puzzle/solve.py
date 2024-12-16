# Feel free to modify the code to facilitate your implementation, e.g., add or modify functions, import other modules.
import argparse
import numpy as np


class Puzzle(object):
    def __init__(self, file_path=None, puzzle=None):
        self.size = 3
        self.goal_puzzle = np.array([[1, 2, 3],[8, 0, 4],[7, 6, 5]])

        if puzzle is not None:
            self.puzzle = puzzle
        elif file_path is not None:
            self.puzzle = self.read_puzzle(file_path)
            assert self.is_solvable(self.puzzle), '8-puzzle has an unsolvable initial state.'
        else:
            self.puzzle = self.make_puzzle()
            
    def make_puzzle(self):
        tiles = range(self.size ** 2)
        tiles = list(tiles)
        np.random.shuffle(tiles)

        while not self.is_solvable(np.array(tiles).reshape((self.size, self.size))):
            np.random.shuffle(tiles)

        return np.array(tiles).reshape((self.size, self.size))
    
    def read_puzzle(self, file_path):
        with open(file_path, 'r') as file:
            puzzle = np.array([list(map(int, line.strip().split())) for line in file.readlines()])
            assert puzzle.shape[0]==self.size and puzzle.shape[1]==self.size, "8-puzzle should have a 3 * 3 board."
        return puzzle

    def is_solvable(self, puzzle):
        # Based on http://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable
        goal_puzzle = self.goal_puzzle.flatten()
        goal_inversions = 0
        for i in range(len(goal_puzzle)):
            for j in range(i+1, len(goal_puzzle)):
                if goal_puzzle[i] > goal_puzzle[j] and goal_puzzle[i] != 0 and goal_puzzle[j] != 0:
                    goal_inversions += 1

        puzzle = puzzle.flatten()
        inversions = 0
        for i in range(len(puzzle)):
            for j in range(i+1, len(puzzle)):
                if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                    inversions += 1

        return inversions % 2 == goal_inversions % 2
    
    def get_successor(self, tile):
        if tile == 1: return 2
        if tile == 2: return 3
        if tile == 3: return 4
        if tile == 4: return 5
        if tile == 5: return 6
        if tile == 6: return 7
        if tile == 7: return 8
        if tile == 8: return 0  # No successor for the last tile
        return 0
    
    def get_neighbors(self):
        neighbors = []
        zero_pos = np.argwhere(self.puzzle == 0)[0]  # Find the position of the empty tile (0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for direction in directions:
            new_x, new_y = zero_pos[0] + direction[0], zero_pos[1] + direction[1]
            if 0 <= new_x < self.size and 0 <= new_y < self.size:  # Check bounds
                new_puzzle = self.puzzle.copy()
                new_puzzle[zero_pos[0], zero_pos[1]], new_puzzle[new_x, new_y] = new_puzzle[new_x, new_y], new_puzzle[zero_pos[0], zero_pos[1]]
                neighbors.append(Puzzle(puzzle=new_puzzle))  # Create a new Puzzle state

        return neighbors

    def misplaced_tiles(self):
        # Implement it if this heuristic is admissible
        return np.sum(self.puzzle != self.goal_puzzle) - 1  # Subtract 1 for the empty tile


    def manhattan_distance(self):
        # Implement it if this heuristic is admissible
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                tile = self.puzzle[i][j]
                if tile != 0:
                    goal_position = np.argwhere(self.goal_puzzle == tile)[0]
                    distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
        return distance

    
    def nilsson_heuristic(self):
        # Implement it if this heuristic is admissible
        P = self.manhattan_distance()
        S = 0
        perimeter_positions = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
        for i in range(len(perimeter_positions) - 1):
            current_tile = self.puzzle[perimeter_positions[i]]
            next_tile = self.puzzle[perimeter_positions[i + 1]]
            if current_tile != 0 and (current_tile + 1) % 8 != next_tile % 8:
                S += 2
        if self.puzzle[1, 1] != 0:
            S += 1
        return P + 3 * S


    def mostow_prieditis_heuristic(self):
        # Implement it if this heuristic is admissible
        out_of_row = np.sum([1 for i in range(1, 9) if np.argwhere(self.puzzle == i)[0][0] != np.argwhere(self.goal_puzzle == i)[0][0]])
        out_of_col = np.sum([1 for i in range(1, 9) if np.argwhere(self.puzzle == i)[0][1] != np.argwhere(self.goal_puzzle == i)[0][1]])
        return out_of_row + out_of_col


def a_star_algorithm(start_puzzle, heuristic):

    open_set = [] # Stores (f(n), state, path)
    initial_h = heuristic(start_puzzle)  
    open_set.append((0 + initial_h, start_puzzle, []))  
    visited = set()  
    num_node_expand = 0 

    while open_set:
        open_set.sort(key=lambda x: x[0])  
        f_n, current_state, path = open_set.pop(0)  

        if np.array_equal(current_state.puzzle, current_state.goal_puzzle):
            return path + [current_state], num_node_expand  

        visited.add(tuple(map(tuple, current_state.puzzle)))  
        num_node_expand += 1

        for next_state in current_state.get_neighbors():
            state_tuple = tuple(map(tuple, next_state.puzzle))  

            if state_tuple not in visited:
                g_n = len(path) + 1  
                h_n = heuristic(next_state)  
                open_set.append((g_n + h_n, next_state, path + [current_state])) 

    return None, num_node_expand  
    

def write_output(name, data, student_id):
    with open(name, 'w') as file:
        file.write(str(student_id) + '\n')
        for state in data:
            for row in state.puzzle:
                file.write(' '.join(map(str, row)) + '\n')
            file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--puzzle", default="puzzle_1.txt", help="Path to txt file containing 8-puzzle")
    parser.add_argument("-H", "--heuristic", type=int, help="Heuristic mode to use. 1: Use Misplaced Tiles; 2: Use Manhattan distance; 3: Use Nilsson Heuristic; 4: Use Mostow and Prieditis Heuristic", default=1, choices=[1, 2, 3, 4]) # You can change the allowable values to only those representing admissible heuristics
    parser.add_argument("-o", "--output_file", default="output_1.txt", help="Path to output txt file")

    args = parser.parse_args()

    if args.puzzle:
        initial_state = Puzzle(args.puzzle)
    else:
        initial_state = Puzzle()

    heuristic_idx = {
    1: "Misplaced Tiles",
    2: "Manhattan Distance",
    3: "Nilsson Heuristic",
    4: "Mostow and Prieditis Heuristic",
    }

    heuristics = {
    "Misplaced Tiles": lambda state: state.misplaced_tiles(),
    "Manhattan Distance": lambda state: state.manhattan_distance(),
    "Nilsson Heuristic": lambda state: state.nilsson_heuristic(),
    "Mostow and Prieditis Heuristic": lambda state: state.mostow_prieditis_heuristic(),
    }
    
    name = heuristic_idx[args.heuristic]
    heuristic = heuristics[name]
    print(f"Using {name}:")
    result_list = a_star_algorithm(initial_state, heuristic)
    if result_list:
        path, num_node_expand = result_list
        print(f"Solution found with {len(path) - 1} moves. {num_node_expand} nodes are expanded.")
        write_output(args.output_file, path, "20966494")
    else:
        print("No solution found.")