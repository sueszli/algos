# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "blessed==1.20.0",
# ]
# ///
import random
import sys
import time

import blessed


def update_state(game, term):
    new_head = move_snake(game)

    # game over conditions
    body = game["snake"][1:]
    has_collision_with_self = new_head in body
    has_collision_with_wall = not (1 <= new_head[0] < term.width - 1 and 1 <= new_head[1] < term.height - 1)
    if has_collision_with_self or has_collision_with_wall:
        return None

    game["snake"].insert(0, new_head)

    handle_fruit_eating(game, term)

    return game


def move_snake(game):
    head_x, head_y = game["snake"][0]
    if game["direction"] == "KEY_LEFT":
        head_x -= 1
    elif game["direction"] == "KEY_RIGHT":
        head_x += 1
    elif game["direction"] == "KEY_UP":
        head_y -= 1
    elif game["direction"] == "KEY_DOWN":
        head_y += 1
    return (head_x, head_y)


def handle_fruit_eating(game, term):
    if game["snake"][0] == game["fruit"]:
        game["score"] += 1
        game["fruit"] = (random.randint(1, term.width - 2), random.randint(1, term.height - 2))
    else:
        game["snake"].pop()


def render(term, game):
    print(term.home + term.clear, end="")

    # draw border
    print(term.move_xy(0, 0) + "┌" + "─" * (term.width - 2) + "┐")
    for y in range(1, term.height - 1):
        print(term.move_xy(0, y) + "│", end="")
        print(term.move_xy(term.width - 1, y) + "│")
    print(term.move_xy(0, term.height - 1) + "└" + "─" * (term.width - 2) + "┘", end="")

    # snake
    for x, y in game["snake"]:
        print(term.move_xy(x, y) + "█", end="")

    # fruit
    print(term.move_xy(game["fruit"][0], game["fruit"][1]) + "ó", end="")
    sys.stdout.flush()


def game_loop(term, game):
    with term.cbreak(), term.hidden_cursor():
        val = ""
        while val.lower() != "q":
            # process input
            val = term.inkey(timeout=0)
            if val.is_sequence:
                game["direction"] = val.name

            # update state
            game = update_state(game, term)
            if game is None:
                break

            # render
            render(term, game)

            time.sleep(0.1)  # game speed

    return game


def main():
    term = blessed.Terminal()
    game = {
        "snake": [(term.width // 2 - i, term.height // 2) for i in range(7)],
        "fruit": (random.randint(1, term.width - 2), random.randint(1, term.height - 2)),
        "direction": "KEY_RIGHT",
        "score": 0,
    }

    final_game_state = game_loop(term, game)

    # "game over" screen
    print(term.home + term.clear)
    score = final_game_state["score"] if final_game_state else 0
    msg = f"Game Over! Score: {score}"
    print(term.move_xy(term.width // 2 - len(msg) // 2, term.height // 2) + msg)


if __name__ == "__main__":
    main()
