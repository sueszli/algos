# assignment:
# read the floor plan below and print the number of chairs per room.
# additionally, sort alphabetically by room name
#
# example output:
# ```
# total:
# W: 3, P: 2, S: 0, C: 0
# living room:
# W: 3, P: 0, S: 0, C: 0
# office:
# W: 0, P: 2, S: 0, C: 0
# ```
#
# also see: https://stackoverflow.com/questions/71826437/parsing-ascii-floor-plan-image-in-python


FLOOR_PLAN = """
+-----------+------------------------------------+
|           |                                    |
| (closet)  |                                    |
|         P |                            S       |
|         P |         (sleeping room)            |
|         P |                                    |
|           |                                    |
+-----------+    W                               |
|           |                                    |
|        W  |                                    |
|           |                                    |
|           +--------------+---------------------+
|                          |                     |
|                          |                W W  |
|                          |    (office)         |
|                          |                     |
+--------------+           |                     |
|              |           |                     |
| (toilet)     |           |             P       |
|   C          |           |                     |
|              |           |                     |
+--------------+           +---------------------+
|              |           |                     |
|              |           |                     |
|              |           |                     |
| (bathroom)   |           |      (kitchen)      |
|              |           |                     |
|              |           |      W   W          |
|              |           |      W   W          |
|       P      +           |                     |
|             /            +---------------------+
|            /                                   |
|           /                                    |
|          /                          W    W   W |
+---------+                                      |
|                                                |
| S                                   W    W   W |
|                (living room)                   |
| S                                              |
|                                                |
|                                                |
|                                                |
|                                                |
+--------------------------+---------------------+
                           |                     |
                           |                  P  |
                           |  (balcony)          |
                           |                 P   |
                           |                     |
                           +---------------------+
"""


import re

FLOOR_PLAN = FLOOR_PLAN.splitlines()
FLOOR_PLAN = [line + " " for line in FLOOR_PLAN][1:]  # regex hack so wall before balcony is detected

CHAIR_TYPES = "WPSC"
queue = []
ans = []

for line in FLOOR_PLAN:
    regex_matches = list(re.finditer(r"[^/\\|+-]+", line))
    regex_matches = [{"start": rm.start(), "end": rm.end(), "text": rm.group()} for rm in regex_matches]
    if not regex_matches:
        continue

    print(f"{line}")
    new_queue = []

    for room in queue:
        rm_in_room = lambda rm, room: max(room["rms"][-1]["start"], rm["start"]) < min(room["rms"][-1]["end"], rm["end"])
        rms = [rm for rm in regex_matches if rm_in_room(rm, room)]
        regex_matches = [rm for rm in regex_matches if not rm_in_room(rm, room)]

        if rms:
            room["rms"].extend(rms)
            new_queue.append(room)
        else:
            text = "".join([rm["text"] for rm in room["rms"]])
            name = re.search(r"\((?:[A-Za-z]+\s?)+\)", text)
            assert name
            room["name"] = name[0][1:-1]
            room["chairs"] = {f: text.count(f) for f in CHAIR_TYPES}
            ans.append(room)
            print("\t" * 6 + f"completed '{room['name']}' with {room['chairs']}")

    queue = new_queue

    for rm in regex_matches:
        queue.append({"name": None, "chairs": {}, "rms": [rm]})
        print("\t" * 5 + f"\tcreated new room ({rm['start']}-{rm['end']})")


def pretty_print(ans):
    assert all(rm["name"] for rm in ans)

    total_chairs = {f: sum(rm["chairs"][f] for rm in ans) for f in CHAIR_TYPES}
    print(f"total:")
    print(", ".join(f"{f}: {total_chairs[f]}" for f in CHAIR_TYPES))

    ans = sorted(ans, key=lambda rm: rm["name"])
    for rm in ans:
        print(f"{rm['name']}:")
        print(", ".join(f"{f}: {rm['chairs'][f]}" for f in CHAIR_TYPES))


pretty_print(ans)
