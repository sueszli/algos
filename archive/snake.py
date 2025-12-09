# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "blessed==1.20.0",
# ]
# ///
# 
#              ____
#             / . .\
#             \  ---<
#              \  /
#    __________/ /
# -=:___________/
# 
# a tiny benchmarking suite for the classic snake game.
# created in collaboration with @pinouchon from tufalabs.ai.
# can you beat my high score?
# 
# usage:
# 
#    $ uv run snake.py
#    $ uv run snake.py --cli --runs 5
# 
# example output (live play feed from solver):
# 
#   ┌Score: 9────────────────────┐
#   │███████████                 │
#   │█                           │
#   │                            │
#   │                            │
#   │     ó                      │
#   └────────────────────────────┘

import argparse
import heapq
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import blessed


# ============================================================================
# state
# ============================================================================


@dataclass(frozen=True)
class GameState:
    snake: Tuple[Tuple[int, int], ...]
    fruit: Tuple[Tuple[int, int]]
    direction: str
    score: int
    term_width: int
    term_height: int


# ============================================================================
# utils
# ============================================================================


def dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    # calculates the manhattan distance between two points
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(game: GameState, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    width = game.term_width - 2
    height = game.term_height - 2

    # the set of nodes already evaluated
    closed_set = set()

    # the set of currently discovered nodes that are not evaluated yet.
    # initially, only the start node is known.
    open_set = [(0, start)]  # (f_score, node)

    # for each node, which node it can most efficiently be reached from.
    # if a node can be reached from many nodes, came_from will eventually contain the most efficient previous step.
    came_from = {}

    # for each node, the cost of getting from the start node to that node.
    g_score = {(x, y): float("inf") for x in range(1, width + 1) for y in range(1, height + 1)}
    g_score[start] = 0

    # for each node, the total cost of getting from the start node to the goal by passing by that node. that value is partly known, partly heuristic.
    f_score = {(x, y): float("inf") for x in range(1, width + 1) for y in range(1, height + 1)}
    f_score[start] = dist(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (1 <= neighbor[0] <= width and 1 <= neighbor[1] <= height):
                continue

            if neighbor in closed_set:
                continue

            # don't run into the snake's body
            # we allow the path to go to the tail, because it will move
            if neighbor in game.snake and neighbor != game.snake[-1]:
                continue

            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + dist(neighbor, goal)
                if (f_score[neighbor], neighbor) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def count_reachable_cells(game: GameState, start: Tuple[int, int]) -> int:
    # counts the number of reachable cells from a starting point using bfs.
    width = game.term_width - 2
    height = game.term_height - 2
    q = [start]
    visited = {start}
    count = 0
    while q:
        cell = q.pop(0)
        count += 1
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (cell[0] + dx, cell[1] + dy)

            if not (1 <= neighbor[0] <= width and 1 <= neighbor[1] <= height):
                continue

            if neighbor in visited:
                continue

            if neighbor in game.snake:
                continue

            visited.add(neighbor)
            q.append(neighbor)
    return count


# ============================================================================
# solver
# ============================================================================


DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = {
    "KEY_UP": (0, -1),
    "KEY_DOWN": (0, 1),
    "KEY_LEFT": (-1, 0),
    "KEY_RIGHT": (1, 0),
}


def _vector_to_direction(vector: Tuple[int, int]) -> Optional[str]:
    for direction, offset in DIRECTION_VECTORS.items():
        if offset == vector:
            return direction
    return None


def _within_bounds(game: GameState, position: Tuple[int, int]) -> bool:
    width = game.term_width - 2
    height = game.term_height - 2
    return 1 <= position[0] <= width and 1 <= position[1] <= height


def _is_move_valid(game: GameState, direction: str) -> bool:
    dx, dy = DIRECTION_VECTORS[direction]
    head_x, head_y = game.snake[0]
    next_head = (head_x + dx, head_y + dy)

    if not _within_bounds(game, next_head):
        return False

    if next_head in game.snake:
        return False

    return True


def _apply_move(game: GameState, direction: str) -> Optional[GameState]:
    if not _is_move_valid(game, direction):
        return None

    dx, dy = DIRECTION_VECTORS[direction]
    head_x, head_y = game.snake[0]
    next_head = (head_x + dx, head_y + dy)
    snake = list(game.snake)

    if next_head == game.fruit:
        snake.insert(0, next_head)
        score = game.score + 1
    else:
        snake.insert(0, next_head)
        snake.pop()
        score = game.score

    return GameState(
        snake=tuple(snake),
        fruit=game.fruit,
        direction=direction,
        score=score,
        term_width=game.term_width,
        term_height=game.term_height,
    )


def _direction_from_path(path: Tuple[Tuple[int, int], ...]) -> Optional[str]:
    if len(path) < 2:
        return None
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return _vector_to_direction((dx, dy))


def _has_escape_route(game: GameState) -> bool:
    # return True if the snake can keep moving safely from the given state

    # ensure that the snake will still be able to reach its tail after the move.
    path_to_tail = a_star_search(game, game.snake[0], game.snake[-1])
    if path_to_tail is not None:
        return True

    # fall back to checking the size of the accessible region to avoid dead ends.
    reachable = count_reachable_cells(game, game.snake[0])
    return reachable > 0


def _simulate_path(game: GameState, path: Sequence[Tuple[int, int]]) -> Optional[GameState]:
    state = game
    for index in range(1, len(path)):
        prev = path[index - 1]
        curr = path[index]
        direction = _vector_to_direction((curr[0] - prev[0], curr[1] - prev[1]))
        if direction is None:
            return None
        state = _apply_move(state, direction)
        if state is None:
            return None

    if state and _has_escape_route(state) and _follows_cycle(state):
        return state
    return None


def _is_path_safe(game: GameState, path: Sequence[Tuple[int, int]]) -> bool:
    if len(path) < 2:
        return False

    final_state = _simulate_path(game, path)
    return final_state is not None


def _follows_cycle(game: GameState) -> bool:
    width = game.term_width - 2
    height = game.term_height - 2
    for idx in range(len(game.snake) - 1):
        current = game.snake[idx]
        nxt = game.snake[idx + 1]
        if _hamilton_successor_map(width, height).get(nxt) != current:
            return False
    return True


OPPOSITE_DIRECTIONS = {
    "KEY_LEFT": "KEY_RIGHT",
    "KEY_RIGHT": "KEY_LEFT",
    "KEY_UP": "KEY_DOWN",
    "KEY_DOWN": "KEY_UP",
}


def _hamilton_rule(x: int, y: int, direction: str, width: int, height: int) -> str:
    if y == 1:
        target_direction = "KEY_LEFT" if x > 1 else "KEY_DOWN"
    elif x == width:
        target_direction = "KEY_UP" if y > 1 else "KEY_LEFT"
    elif y == height:
        target_direction = "KEY_RIGHT" if x % 2 != 0 else "KEY_UP"
    elif x % 2 != 0:
        target_direction = "KEY_DOWN"
    elif y > 2:
        target_direction = "KEY_UP"
    else:
        target_direction = "KEY_RIGHT"

    if OPPOSITE_DIRECTIONS[direction] == target_direction:
        return "KEY_DOWN" if y + 1 <= height else "KEY_UP"
    return target_direction


@lru_cache(maxsize=None)
def _hamilton_successor_map(width: int, height: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    assert width % 2 == 0, "width must be even for this Hamiltonian cycle."

    successors: Dict[Tuple[int, int], Tuple[int, int]] = {}
    x, y = width, 1
    direction = "KEY_LEFT"

    start = (x, y)
    while True:
        move = _hamilton_rule(x, y, direction, width, height)
        dx, dy = DIRECTION_VECTORS[move]
        nx, ny = x + dx, y + dy
        successors[(x, y)] = (nx, ny)
        x, y = nx, ny
        direction = move
        if (x, y) == start:
            break

    return successors


def _hamilton_direction(game: GameState) -> Optional[str]:
    width = game.term_width - 2
    height = game.term_height - 2
    head = game.snake[0]
    successor_map = _hamilton_successor_map(width, height)
    next_cell = successor_map.get(head)
    if next_cell is None:
        return None

    dx = next_cell[0] - head[0]
    dy = next_cell[1] - head[1]
    return _vector_to_direction((dx, dy))


def get_next_direction(game: GameState) -> Optional[str]:
    head = game.snake[0]

    path_to_fruit = a_star_search(game, head, game.fruit)
    if path_to_fruit and len(path_to_fruit) > 1:
        direction = _direction_from_path(tuple(path_to_fruit))
        if direction and _is_move_valid(game, direction) and _is_path_safe(game, path_to_fruit):
            return direction

    return _hamilton_direction(game)


# ============================================================================
# game
# ============================================================================


def update_game_state(game: GameState) -> Optional[GameState]:
    direction = get_next_direction(game)
    if direction is None:
        return None

    illegal_turns = {"KEY_LEFT": "KEY_RIGHT", "KEY_RIGHT": "KEY_LEFT", "KEY_UP": "KEY_DOWN", "KEY_DOWN": "KEY_UP"}
    if illegal_turns.get(game.direction) == direction:
        direction = game.direction

    head_x, head_y = game.snake[0]
    if direction == "KEY_LEFT":
        head_x -= 1
    elif direction == "KEY_RIGHT":
        head_x += 1
    elif direction == "KEY_UP":
        head_y -= 1
    elif direction == "KEY_DOWN":
        head_y += 1
    new_head = (head_x, head_y)

    if new_head in game.snake or not (1 <= new_head[0] < game.term_width - 1 and 1 <= new_head[1] < game.term_height - 1):
        return None

    new_snake = list(game.snake)
    new_snake.insert(0, new_head)

    if new_head == game.fruit:
        new_score = game.score + 1
        all_cells = set((x, y) for x in range(1, game.term_width - 1) for y in range(1, game.term_height - 1))
        valid_fruit_cells = all_cells - set(new_snake)
        if not valid_fruit_cells:
            return GameState(tuple(new_snake), new_head, direction, new_score, game.term_width, game.term_height)
        new_fruit = random.choice(list(valid_fruit_cells))

        return GameState(tuple(new_snake), new_fruit, direction, new_score, game.term_width, game.term_height)
    else:
        new_snake.pop()
        return GameState(tuple(new_snake), game.fruit, direction, game.score, game.term_width, game.term_height)


def render(term: blessed.Terminal, game: GameState):
    print(term.home + term.clear, end="")

    print(term.move_xy(0, 0) + "┌" + "─" * (game.term_width - 2) + "┐")
    for y in range(1, game.term_height - 1):
        print(term.move_xy(0, y) + "│", end="")
        print(term.move_xy(game.term_width - 1, y) + "│")
    print(term.move_xy(0, game.term_height - 1) + "└" + "─" * (game.term_width - 2) + "┘", end="")

    score_text = f"Score: {game.score}"
    print(term.move_xy(1, 0) + score_text, end="")

    for x, y in game.snake:
        print(term.move_xy(x, y) + "█", end="")

    print(term.move_xy(game.fruit[0], game.fruit[1]) + "ó", end="")
    sys.stdout.flush()


def game_loop(term: blessed.Terminal, initial_game_state: GameState):
    game = initial_game_state
    last_game_state = initial_game_state
    with term.cbreak(), term.hidden_cursor():
        while game:
            render(term, game)
            last_game_state = game
            game = update_game_state(game)
            time.sleep(0.01)
    return last_game_state


def cli_game_loop(initial_game_state: GameState) -> Tuple[GameState, int]:
    game = initial_game_state
    steps = 0
    last_game_state = initial_game_state
    steps_since_last_fruit = 0

    while game:
        last_game_state = game
        prev_score = game.score
        game = update_game_state(game)
        steps += 1

        if not game:
            break

        if game.score > prev_score:
            steps_since_last_fruit = 0
        else:
            steps_since_last_fruit += 1

        live_lock = steps_since_last_fruit > (initial_game_state.term_width - 2) * (initial_game_state.term_height - 2) * 2
        if live_lock:
            break
    return last_game_state, steps


def init_game_state(term_width: int, term_height: int) -> GameState:
    snake = tuple((term_width // 2 - i, term_height // 2) for i in range(3))
    all_cells = set((x, y) for x in range(1, term_width - 1) for y in range(1, term_height - 1))
    valid_fruit_cells = all_cells - set(snake)
    fruit = random.choice(list(valid_fruit_cells))
    return GameState(snake=snake, fruit=fruit, direction="KEY_RIGHT", score=0, term_width=term_width, term_height=term_height)


def main():
    parser = argparse.ArgumentParser(description="run the snake game.")
    parser.add_argument("--cli", action="store_true", help="run in cli mode without graphics.")
    parser.add_argument("--runs", type=int, default=1, help="number of times to run the game.")
    args = parser.parse_args()

    # silent mode for benchmarking
    if args.cli:
        total_score = 0
        total_steps = 0
        term_width = 10
        term_height = 10
        max_score = float((term_width - 2) * (term_height - 2) - 3)
        for _ in range(args.runs):
            initial_game_state = init_game_state(term_width, term_height)
            final_game_state, steps = cli_game_loop(initial_game_state)
            total_score += final_game_state.score
            total_steps += steps
            assert final_game_state.score == max_score, f"incorrect solution: only got {final_game_state.score}/{max_score}"
        print(f"average steps: {total_steps / args.runs}")
        exit(0)

    # graphical mode for debugging
    term = blessed.Terminal()
    initial_game_state = init_game_state(term.width, term.height)
    final_game_state = game_loop(term, initial_game_state)
    print(term.home + term.clear)
    score = final_game_state.score
    max_size = (final_game_state.term_width - 2) * (final_game_state.term_height - 2)
    if len(final_game_state.snake) >= max_size:
        msg = "you won!"
    else:
        msg = f"game over! score: {score}"
    print(term.move_xy(term.width // 2 - len(msg) // 2, term.height // 2) + msg)


if __name__ == "__main__":
    main()
