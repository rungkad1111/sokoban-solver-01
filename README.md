# Sokoban solver — room decomposition + push-optimal A*

A Sokoban solver for make.fun's sokoban-solver-01. Every level the competition
draws is a one-wide hallway with one to four rooms off it, each room holding
3 or 4 boxes and as many goals behind a single door. A box pushed through a
door into the hallway can never come back, so each box stays in its room and
the level's push optimum is the sum of its rooms' optima.

The solver splits the level into rooms (the hallway's squares are dead — a
box there reaches no goal — so cutting them separates the rooms), then solves
each room push-optimally with an A* search over push states: the state is the
sorted box squares plus the player's reachable region, the heuristic is the
min-cost assignment of boxes to goals over single-box push-distance tables,
and dead squares plus freeze deadlocks prune the search. Per-room fallbacks
(uniform-cost, then greedy) and a whole-level fallback keep every instance
answered. All budgets are counted in expansions, never in time, so two runs
write byte-identical answers.

Run it as the grader does:

```
node main.ts   # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Locally, `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` stand in for the mounts.
