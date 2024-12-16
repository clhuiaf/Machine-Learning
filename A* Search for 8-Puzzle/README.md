# A* Search for 8-Puzzle - README

## Overview
This project implements an A* search algorithm to solve the 8-puzzle problem. The program allows the user to specify different admissible heuristics to evaluate the best path to the goal state.

## Dependencies
- Python 3.x
- NumPy library

You can install NumPy using pip if you haven't already:

```bash
pip install numpy

## Files included
20966494_problem1.py: The main Python script containing the implementation of the A* search algorithm.
puzzle_1.txt: An example input file containing the initial state of the 8-puzzle.
output_1.txt: The output file where the solution will be written (this file will be created by the program).
dependencies_20966494.md: This documentation file.

## How to Run the Code
Prepare the Input File: Create a text file (e.g., puzzle_1.txt) with the initial configuration of the 8-puzzle. The format should be a 3x3 grid of numbers where 0 represents the empty space.

Example of puzzle_1.txt:

1 2 3
7 8 4
6 5 0

Run the Program: Open a terminal/command prompt and navigate to the directory where your Python script is located. Run the script using the following command:

python 20966494_problem1.py -p puzzle_1.txt -H 1 -o output_1.txt

-p: Path to the input file containing the puzzle (default is puzzle_1.txt).
-H: Heuristic mode to use (default is 1, which corresponds to Misplaced Tiles).
1: Misplaced Tiles
2: Manhattan Distance
3: Nilsson Heuristic
4: Mostow and Prieditis Heuristic
-o: Path to the output file where the solution will be saved (default is output_1.txt).
Check the Output: After running the program, check the output_1.txt file for the solution path from the initial state to the goal state. The first line will contain your student ID, followed by the sequence of puzzle states.

Example Command
python YourStudentID_problem1.py -p puzzle_1.txt -H 2 -o output_1.txt

Important Notes
Ensure that the input puzzle is solvable. The program will assert that the initial state is solvable before proceeding.
Modify the input file path and heuristic as needed based on your requirements.
Contact
For any questions or issues, please reach out to your course instructor or teaching assistant.

This README provides a clear guide on how to set up and run your code, along with deta
