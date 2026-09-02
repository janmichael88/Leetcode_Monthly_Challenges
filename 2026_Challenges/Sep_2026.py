################################################
# 3568. Minimum Moves to Clean the Classroom
# 02SEP26
################################################
class Solution:
    def minMoves(self, classroom: List[str], energy: int) -> int:
        '''
        notice there are most 10 L cell in thr grid
        bfs with states (x,y,mask,e,steps)
             initializing with (sx, sy, 0, energy, 0), and for each move update e (–1 per step), update mask on 'L', reset e=energy on 'R', and return steps when mask == fullMask.
        maintain 3d array: bestEnergy[x][y][mask] and skip any new state with e <= bestEnergy[x][y][mask]
            since a smaller enery wouldn't allow for exploraton
        '''
        rows, cols = len(classroom), len(classroom[0])

        sx, sy = -1, -1
        litter = []

        for i in range(rows):
            for j in range(cols):
                if classroom[i][j] == "S":
                    sx, sy = i, j
                elif classroom[i][j] == "L":
                    litter.append((i, j))

        litter = {coord: i for i, coord in enumerate(litter)}

        #full_mask = (1 << len(litter)) - 1

        # (row, col, mask) -> maximum energy we've reached this state with
        best_energy = defaultdict(lambda: float('-inf'))

        starting_state = (sx, sy, set(), energy, 0)
        q = deque([starting_state])

        #init first state
        best_energy[(sx, sy, 0)] = energy

        while q:
            x, y, curr_mask, curr_energy, steps = q.popleft()

            if len(curr_mask) == len(litter):
                return steps

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue

                if classroom[nx][ny] == "X":
                    continue

                # Moving costs one energy
                neigh_energy = curr_energy - 1

                if neigh_energy < 0:
                    continue

                neigh_mask = curr_mask.copy()

                # Recharge
                if classroom[nx][ny] == "R":
                    neigh_energy = energy

                # Collect litter
                elif classroom[nx][ny] == "L":
                    neigh_mask.add((nx,ny))

                state_key = (nx, ny, frozenset(neigh_mask))

                # Only explore if we arrive with more energy
                if neigh_energy > best_energy[state_key]:
                    best_energy[state_key] = neigh_energy
                    q.append((nx, ny, neigh_mask, neigh_energy, steps + 1))

        return -1