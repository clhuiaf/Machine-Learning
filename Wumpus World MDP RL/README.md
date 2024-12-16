In this problem, we will delve into the application of Markov Decision Processes (MDPs) 
and the Value Iteration and Policy Iteration in the context of the Wumpus World. 
The Wumpus World is a well-known problem in the field of artificial intelligence, 
where the objective is to control an agent as it navigates through a grid-like environment 
in search of gold, all while avoiding deadly pits and the formidable Wumpus creature.

As you observed, the agent starts at the grid coordinate x = 0, y = 0, (x is the horizontal axes, y is the vertical axes) and its objectives are the following:
- Finding the gold, which provides a significant positive reward (+10).
- Avoiding the pits and the Wumpus, which are associated with negative penalties (-5 for each pit and -10 for the Wumpus).
- Minimizing the incurred movement penalty (-0.4 for each non-goal cell). Due to the noise of the control signal, the movements are stochastic: There is an 80% chance that the agent moves in the intended direction. To be more specific, there is a 10% chance that the agent moves in one of the orthogonal directions. For example, if the agent intends to move UP, there’s an 80% chance it will move UP, a 10% chance it will move LEFT, and a 10% chance it will move RIGHT.

There are three user-defined parameters in the program, namely, ‘gamma’, ‘eta’ and ‘max_iter’. The usages are the following:
- gamma: sets a discount factor of future rewards. It represents how much future rewards are valued compared to immediate rewards.
- eta: sets a threshold for the maximum value error between two adjacent iterations to assess algorithm convergence. If the maximum value error is less than this threshold, the iteration process is terminated.
- max_iter: sets the maximum number of iterations that the implemented algorithm will run.
