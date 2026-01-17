from dotenv import load_dotenv
load_dotenv(override = True)

from service import *
from schema import *

import time

def test_deepseek_api_block():
    request = ChatCompletionRequest(
        model = "",
        messages = [
            Message(
                role = "user",
                content = "Predict the exact and complete standard output (stdout) of the following Python program, including every single print statement. The program contains several injected debug print statements starting with '__DBG_'. You must include these in your prediction exactly as they would appear in the output, along with any other output the program produces. Program: ```python from collections import defaultdict import sys input = sys.stdin.readline print('__DBG_0__', repr(input) if isinstance(input, (int, float, str, bool, type(None))) else type(input).__name__) class graph: def __init__(self, n, mark): self.d = defaultdict(list) self.n = n self.mark = mark print('__DBG_1__', repr(n) if isinstance(n, (int, float, str, bool, type(None))) else type(n).__name__) def add(self, s, d): self.d[s].append(d) self.d[d].append(s) def bfs(self, s, dis): marked = s print('__DBG_2__', repr(marked) if isinstance(marked, (int, float, str, bool, type(None))) else type(marked).__name__) visited = [False] * self.n visited[s] = True print('__DBG_3__', repr(s) if isinstance(s, (int, float, str, bool, type(None))) else type(s).__name__, repr(self) if isinstance(self, (int, float, str, bool, type(None))) else type(self).__name__, repr(input) if isinstance(input, (int, float, str, bool, type(None))) else type(input).__name__) q = [s] print('__DBG_4__', repr(marked) if isinstance(marked, (int, float, str, bool, type(None))) else type(marked).__name__) while q: s = q.pop(0) if s in mark: marked = s print('__DBG_5__', repr(q) if isinstance(q, (int, float, str, bool, type(None))) else type(q).__name__, repr(self) if isinstance(self, (int, float, str, bool, type(None))) else type(self).__name__, repr(input) if isinstance(input, (int, float, str, bool, type(None))) else type(input).__name__) for i in self.d[s]: if visited[i] == False: q.append(i) visited[i] = True dis[i] += dis[s] + 1 return marked n, m, k = map(int, input().split()) mrk = [int(x) for x in input().split()] mark = {} for i in mrk: mark[i - 1] = 1 g = graph(n, mark) for i in range(n - 1): a, b = map(int, input().split()) g.add(a - 1, b - 1) dis = [0] * n u = g.bfs(0, dis) dis = [0] * n d = g.bfs(u, dis) temp = [0] * n x = g.bfs(d, temp) count = 0 for i in range(n): if temp[i] <= k and dis[i] <= k: count += 1 print(count) ``` Input (stdin): ``` 6 2 3 1 2 1 5 2 3 3 4 4 5 5 6 ``` Provide the full stdout content. Do not provide any explanations or commentary outside of the predicted output."
            )
        ],
        stream = False
    )
    response = call_deepseek_api(request)
    print(response.model_dump_json())

def test_openai_api_block():
    request = ChatCompletionRequest(
        model = "",
        messages = [
            Message(
                role = "system",
                content = """
You are playing clobber. # Game Rules CLOBBER RULES: Board: Rectangular grid (5×5, 6×6, or 7×7) filled with alternating black and white pieces. Goal: Be the last player able to move. Movement: On your turn, move one of your pieces orthogonally (horizontally or vertically) to capture an adjacent opponent piece. The captured piece is removed and replaced by your piece. Must capture: Every move must capture an opponent piece. No non-capturing moves allowed. Move Format: Moves are specified as "row col" (e.g., "2 3" means row 2, column 3). Important: You can ONLY move to a position occupied by an opponent piece that is directly adjacent (up/down/left/right) to one of your pieces. Losing: If you have no legal moves (no adjacent opponent pieces to capture), you lose. # Output Format You must respond with ONLY the action ID (a single number). Do NOT include descriptions or explanations. Examples: - For action "0 -> roll": respond "0" - For action "89 -> a3": respond "89"

## Response Format
Return JSON like: {
    "think": <your thinking process about the best action. start with "We must choose the best action from the current state...">,
    "action": <the chosen legal action ID from the given list>
}
"""
            ),
            Message(
                role = "user",
                content = """
Current State:
5oxoxo
4xoxox
3oxoxo
2xoxox
1oxoxo
 abcde


You are Player 0.
Legal Actions:
1 -> a5b5
2 -> a5a4
9 -> c5d5
10 -> c5c4
11 -> c5b5
18 -> e5e4
19 -> e5d5
24 -> b4b5
25 -> b4c4
26 -> b4b3
27 -> b4a4
32 -> d4d5
33 -> d4e4
34 -> d4d3
35 -> d4c4
40 -> a3a4
41 -> a3b3
42 -> a3a2
48 -> c3c4
49 -> c3d3
50 -> c3c2
51 -> c3b3
56 -> e3e4
58 -> e3e2
59 -> e3d3
64 -> b2b3
65 -> b2c2
66 -> b2b1
67 -> b2a2
72 -> d2d3
73 -> d2e2
74 -> d2d1
75 -> d2c2
80 -> a1a2
81 -> a1b1
88 -> c1c2
89 -> c1d1
91 -> c1b1
96 -> e1e2
99 -> e1d1

Your choice (ID only):
"""
            )
        ],
        stream = False
    )
    response = call_openai_api(request, "gpt-5-mini")
    print(response.model_dump_json())

if __name__ == "__main__":
    tick = time.time()

    #test_deepseek_api_block()
    test_openai_api_block()

    delay = time.time() - tick
    print(f"⏰ total delay = {delay:.2f}s")
