# agents/fixed_policy_agent.py
import numpy as np
from collections import deque
from agents.agent import Agent

# Ações (consistentes com environment.ACTION_TO_DELTA)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = {
    UP:    (0, -1),
    RIGHT: (1, 0),
    DOWN:  (0, 1),
    LEFT:  (-1, 0)
}

class FixedPolicyAgent(Agent):
    """
    Agente robusto baseado em exploração local + backtracking.
    - Mantém stack de caminho (path_stack) com posições visitadas.
    - Prefere vizinhos não visitados; faz backtrack quando necessário.
    - Usa delta_y_norm (index 11) como heurística para priorizar movimentos verticalmente em direção ao farol.
    - Debug activável para ver decisões.
    """

    def __init__(self, id: str, sensores: bool=True, debug: bool=False):
        super().__init__(id, politica="fixed", sensores=sensores)
        self.last_action = UP
        self.path_stack = []           # stack com a sequência de posições (x,y)
        self.visited = set()           # conjunto de posições visitadas
        self.debug = debug
        self.steps_since_progress = 0  # contador para detectar stalling

    # --------------------------
    # Helpers
    # --------------------------
    def _get_ranges(self):
        if self.last_observation is None:
            return np.zeros(8, dtype=np.float32)
        return np.array(self.last_observation[:8], dtype=np.float32)

    def _get_delta_y(self):
        try:
            return float(self.last_observation[11])
        except Exception:
            return 0.0

    def _adjacent_pos(self, pos, action):
        x, y = pos
        dx, dy = ACTION_DELTAS[action]
        return (x + dx, y + dy)

    def _action_to_move_towards(self, from_pos, to_pos):
        # devolve a acção que leva de from_pos para to_pos, assumindo adjacência
        fx, fy = from_pos
        tx, ty = to_pos
        dx = tx - fx
        dy = ty - fy
        for a, (adx, ady) in ACTION_DELTAS.items():
            if (adx, ady) == (dx, dy):
                return a
        return None

    def livre(self, action):
        ranges = self._get_ranges()
        # índice 0..3 correspondem N,E,S,W (front/back/left/right)
        return float(ranges[action]) > 0.0

    # --------------------------
    # Core decision helpers
    # --------------------------
    def _ensure_position_on_stack(self):
        pos = getattr(self, "posicao", None)
        if pos is None:
            return
        if not self.path_stack:
            # Start: push
            self.path_stack.append(pos)
            self.visited.add(pos)
            return
        # If moved to new pos not equal to top -> push or adjust
        top = self.path_stack[-1]
        if pos != top:
            # if moved back to previous (backtrack)
            if len(self.path_stack) >= 2 and pos == self.path_stack[-2]:
                # pop top (we backtracked)
                self.path_stack.pop()
            else:
                # forward move to new cell
                self.path_stack.append(pos)
                self.visited.add(pos)

    def _unvisited_free_neighbors(self):
        pos = getattr(self, "posicao", None)
        if pos is None:
            return []
        ranges = self._get_ranges()
        neighbors = []
        for a in (UP, RIGHT, DOWN, LEFT):
            if ranges[a] > 0.0:
                adj = self._adjacent_pos(pos, a)
                if adj not in self.visited:
                    neighbors.append((a, adj, ranges[a]))
        return neighbors

    def _preferred_order(self, candidates):
        """
        Ordena candidatos por heurística:
        1) preferir ação que se alinha com delta_y (aproxima ao farol verticalmente)
        2) depois por maior range (mais espaço)
        candidates: list[(action, adj_pos, range_val)]
        """
        delta_y = self._get_delta_y()
        # preferencia vertical: se delta_y < 0 -> prefer UP; if >0 -> prefer DOWN
        preferred = None
        if delta_y < 0:
            preferred = UP
        elif delta_y > 0:
            preferred = DOWN

        def score(c):
            a, _, r = c
            s = 0.0
            if a == preferred:
                s += 10.0
            s += float(r)  # mais espaço melhor
            # evitar reverso imediato (penalizar)
            reverse = (int(self.last_action) + 2) % 4
            if a == reverse:
                s -= 5.0
            return -s  # menor para sort -> queremos maior primeiro, so -s

        return sorted(candidates, key=score)

    # --------------------------
    # Decision: choose action
    # --------------------------
    def age(self) -> int:
        # update path stack based on agent.posicao (env updates posicao)
        self._ensure_position_on_stack()

        # defensive init
        if self.last_observation is None:
            return UP

        ranges = self._get_ranges()

        # detect many adjacent walls -> likely corridor/maze
        paredes_adj = int(np.sum(ranges == 0.0))

        # 1) Try to move to any unvisited free neighbor (DFS preference)
        candidates = self._unvisited_free_neighbors()
        if candidates:
            ordered = self._preferred_order(candidates)
            # choose best candidate action
            action = ordered[0][0]
            self.last_action = int(action)
            self.steps_since_progress = 0
            if self.debug:
                print(f"[MOVE->UNVISITED] pos={self.posicao} act={action} candidates={[(c[0],c[2]) for c in candidates]}")
            return int(action)

        # 2) No unvisited neighbors: backtrack if possible (pop stack)
        if len(self.path_stack) >= 2:
            # target is previous position
            curr = self.path_stack[-1]
            prev = self.path_stack[-2]
            back_action = self._action_to_move_towards(curr, prev)
            if back_action is not None and self.livre(back_action):
                self.last_action = int(back_action)
                self.steps_since_progress += 1
                if self.debug:
                    print(f"[BACKTRACK] pos={self.posicao} back={back_action}")
                return int(back_action)
            # else fallthrough to other strategies

        # 3) If in an open area (few walls) try go-to-goal heuristics
        if paredes_adj < 5:
            # try vertical preference using delta_y_norm
            delta_y = self._get_delta_y()
            desired = None
            if delta_y < 0 and ranges[UP] > 0.0:
                desired = UP
            elif delta_y > 0 and ranges[DOWN] > 0.0:
                desired = DOWN
            else:
                # horizontal by space
                if ranges[RIGHT] > ranges[LEFT] and ranges[RIGHT] > 0.0:
                    desired = RIGHT
                elif ranges[LEFT] > 0.0:
                    desired = LEFT

            if desired is not None and self.livre(desired):
                self.last_action = int(desired)
                self.steps_since_progress = 0
                if self.debug:
                    print(f"[GO-TO-GOAL] pos={self.posicao} desired={desired} delta_y={delta_y:.3f}")
                return int(desired)

        # 4) If many walls (corridor) or no go-to-goal possible -> wall following
        # prefer right-hand rule but avoid immediate reverse unless necessary
        # prefer directions in order [right, forward, left, back]
        d = int(self.last_action) % 4
        order = [ (d + 1) % 4, d, (d - 1) % 4, (d + 2) % 4 ]
        for a in order:
            if self.livre(a):
                self.last_action = int(a)
                if self.debug:
                    print(f"[WALL-FOLLOW/FALLBACK] pos={self.posicao} pick={a} paredes_adj={paredes_adj}")
                return int(a)

        # 5) ultimate fallback: any free direction (search full)
        for a in (UP, RIGHT, DOWN, LEFT):
            if self.livre(a):
                self.last_action = int(a)
                if self.debug:
                    print(f"[ULT-FALLBACK] pos={self.posicao} pick={a}")
                return int(a)

        # 6) no move possible - stay (shouldn't happen)
        if self.debug:
            print(f"[STAY] pos={self.posicao} no moves")
        return int(self.last_action)