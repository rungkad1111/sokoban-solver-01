// sokoban-solver: a push-optimal Sokoban solver built on room decomposition.
//
// Every level this competition draws is a one-wide hallway with one to four
// rooms off it, each room holding 3 or 4 boxes and as many goals, joined to the
// hallway by a single door. A box pushed through a door into the hallway can
// never be pushed back, so each box stays in its own room and the level's
// push optimum is the sum of the rooms' optima. This solver therefore:
//
//   1. splits the level into rooms (the hallway's squares are dead — a box
//      there reaches no goal — so cutting them separates the rooms),
//   2. solves each room push-optimally with an A* search over pushes, the
//      other rooms' boxes standing as walls, the player walking on from
//      where the previous room left them,
//   3. falls back, per room, to a uniform-cost push search and then to a
//      greedy search, so a room yields a valid solution whenever one is
//      findable inside the work budgets, and to a whole-level search when
//      the level is not a clean set of rooms.
//
// The per-room A* state is the sorted box squares plus the player's
// reachable region (named by its lowest square). Its heuristic is the
// min-cost assignment of boxes to goals over single-box push distances
// (reverse-pull BFS tables), which is admissible, so the first solution
// found is push-optimal. Dead squares and freeze deadlocks (a box that can
// never move again, off its goal) prune the search. All budgets are counted
// in expansions, never in time, and every tie is broken by a fixed rule, so
// two runs of this program write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four directions, always in this order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];
const INF = 1e9;

// Work budgets, in search expansions (never the clock).
const ROOM_BUDGETS = [
  ['astar', 600_000],
  ['bfs', 2_000_000],
  ['greedy', 2_000_000],
];
const LEVEL_BUDGETS = [
  ['astar', 1_500_000],
  ['bfs', 3_000_000],
  ['greedy', 3_000_000],
];

/** A level text (XSB) read into walls, goals, boxes and the player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever the text shows: nothing leaves the board,
  // and index arithmetic below can then trust that a step off the floor lands on a wall.
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * pushDist[s]: the fewest pushes that move one box from s to `goal` on the static
 * board (walls only, the other boxes ignored), or INF when no push sequence can.
 * A breadth-first search backwards from the goal: a box now on t was pushed there
 * from s = t - d by a player standing on s - d, so both must be open floor.
 */
function pushDistances(w, wall, goal) {
  const dist = new Int32Array(wall.length).fill(INF);
  dist[goal] = 0;
  const queue = [goal];
  for (let qi = 0; qi < queue.length; qi++) {
    const t = queue[qi];
    for (let d = 0; d < 4; d++) {
      const delta = DY[d] * w + DX[d];
      const s = t - delta;
      const stand = s - delta;
      if (wall[s] || wall[stand] || dist[s] !== INF) continue;
      dist[s] = dist[t] + 1;
      queue.push(s);
    }
  }
  return dist;
}

/**
 * Marks every square the player reaches from `from` (walls and `occupied` in the
 * way) with `stamp` in `seen`, and returns the lowest such square: the canonical
 * name of the player's region, since only the region matters for what can be pushed.
 */
function floodRegion(w, wall, occupied, from, seen, stamp, stack) {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    let q = p - w;
    if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
    q = p + w;
    if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
    q = p - 1;
    if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
    q = p + 1;
    if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
      seen[q] = stamp;
      stack[n++] = q;
    }
  }
  return low;
}

/**
 * Whether the box configuration is a freeze deadlock: a box is frozen when, along
 * both axes, at least one side is blocked by a wall or an already-frozen box (such
 * a box can never be pushed again: the blocked side is neither a destination it
 * can enter nor a square the player can stand on). The frozen set is grown to a
 * fixed point; a frozen box off its goal can never be solved away.
 */
function freezeDeadlock(w, wall, occupied, boxes, goalMask) {
  const frozen = new Set();
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen.has(b)) continue;
      const blocked = (q) => wall[q] === 1 || (occupied[q] === 1 && frozen.has(q));
      const acrossBlocked = blocked(b - 1) || blocked(b + 1);
      const downBlocked = blocked(b - w) || blocked(b + w);
      if (acrossBlocked && downBlocked) {
        frozen.add(b);
        changed = true;
      }
    }
  }
  for (const b of frozen) if (!goalMask[b]) return true;
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in walk letters. */
function walkMoves(w, wall, occupied, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Int8Array(wall.length).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let d = 0; d < 4; d++) {
      const q = p + DY[d] * w + DX[d];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = WALK[how[p]] + s;
  return s;
}

/**
 * A push search on one board: A* ('astar'), uniform cost ('bfs') or greedy
 * best-first ('greedy') over push states, giving up after `budget` expansions.
 * Returns { moves, pushes, endPlayer } for the first goal state reached (for
 * 'astar' and 'bfs' that state is push-optimal), or null.
 */
function searchPushes(w, wall, goals, startBoxes, startPlayer, budget, mode) {
  const size = wall.length;
  const goalMask = new Uint8Array(size);
  for (const g of goals) goalMask[g] = 1;
  const distTables = goals.map((g) => pushDistances(w, wall, g));
  // A square is dead when no goal's table reaches it: a box pushed there is lost.
  const dead = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    if (goalMask[s]) continue;
    let live = false;
    for (const dt of distTables) {
      if (dt[s] < INF) {
        live = true;
        break;
      }
    }
    if (!live) dead[s] = 1;
  }
  if (startBoxes.some((b) => dead[b])) return null;

  // The heuristic: min-cost assignment of boxes to goals over the push-distance
  // tables (exact subset DP for up to 7 boxes, a sum of per-box minima beyond).
  // Admissible: each box needs at least its table distance in pushes of its own,
  // and one push moves one box. Memoised per box configuration.
  const n = startBoxes.length;
  const popcount = new Uint8Array(1 << Math.min(n, 7));
  for (let m = 1; m < popcount.length; m++) popcount[m] = popcount[m >> 1] + (m & 1);
  const hCache = new Map();
  const heuristic = (boxes) => {
    const cacheKey = boxes.join(',');
    const hit = hCache.get(cacheKey);
    if (hit !== undefined) return hit;
    let value;
    if (n <= 7) {
      const full = (1 << n) - 1;
      const dp = new Array(full + 1).fill(INF);
      dp[0] = 0;
      for (let mask = 1; mask <= full; mask++) {
        const bi = popcount[mask] - 1;
        let best = INF;
        for (let j = 0; j < n; j++) {
          if (!(mask & (1 << j))) continue;
          const c = distTables[j][boxes[bi]];
          if (c >= INF) continue;
          const prev = dp[mask ^ (1 << j)] + c;
          if (prev < best) best = prev;
        }
        dp[mask] = best;
      }
      value = dp[full];
    } else {
      let sum = 0;
      value = INF;
      for (const b of boxes) {
        let m = INF;
        for (let j = 0; j < n; j++) if (distTables[j][b] < m) m = distTables[j][b];
        if (m >= INF) {
          sum = INF;
          break;
        }
        sum += m;
      }
      if (sum < INF) value = sum;
    }
    hCache.set(cacheKey, value);
    return value;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;

  // The states, in the order created: boxes, the square the player stands on,
  // the parent state and the push that made this one (box from, box to, direction).
  const stBoxes = [startBoxes.slice()];
  const stPlayer = [startPlayer];
  const stParent = [-1];
  const stFrom = [-1];
  const stTo = [-1];
  const stDir = [-1];
  const stG = [0];
  const bestG = new Map();
  const keyOf = (boxes, low) => low + '|' + boxes.join(',');
  for (const b of startBoxes) occupied[b] = 1;
  const startLow = floodRegion(w, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  bestG.set(keyOf(startBoxes, startLow), 0);

  // The open list: a binary heap on (f, then deeper g, then insertion order).
  const heap = [];
  const hLess = (a, b) => {
    if (a.f !== b.f) return a.f < b.f;
    if (a.g !== b.g) return a.g > b.g;
    return a.seq < b.seq;
  };
  const hPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!hLess(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && hLess(heap[c + 1], heap[c])) c++;
        if (!hLess(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  const fOf = (g, h) => (mode === 'bfs' ? g : mode === 'greedy' ? h : g + h);
  let seq = 0;
  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;
  hPush({ f: fOf(0, h0), g: 0, seq: seq++, state: 0 });

  let expansions = 0;
  let found = -1;
  if (startBoxes.every((b) => goalMask[b])) found = 0;
  while (found < 0 && heap.length > 0) {
    const { g, state } = hPop();
    const boxes = stBoxes[state];
    for (const b of boxes) occupied[b] = 1;
    const low = floodRegion(w, wall, occupied, stPlayer[state], seen, ++stamp, stack);
    const regionStamp = stamp;
    if (bestG.get(keyOf(boxes, low)) !== g) {
      for (const b of boxes) occupied[b] = 0;
      continue; // a stale heap entry: this state was already reached more cheaply
    }
    if (boxes.every((b) => goalMask[b])) {
      found = state;
      for (const b of boxes) occupied[b] = 0;
      break;
    }
    if (++expansions > budget) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const delta = DY[d] * w + DX[d];
        const stand = b - delta;
        const dest = b + delta;
        if (seen[stand] !== regionStamp) continue; // the player cannot stand there
        if (wall[dest] || occupied[dest] || dead[dest]) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        occupied[b] = 0;
        occupied[dest] = 1;
        const frozen = freezeDeadlock(w, wall, occupied, next, goalMask);
        let nextLow = -1;
        if (!frozen) {
          // A separate stamp array: this flood must not erase the current region's marks.
          nextLow = floodRegion(w, wall, occupied, b, mark, ++markStamp, stack);
        }
        occupied[dest] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const key = keyOf(next, nextLow);
        const ng = g + 1;
        const known = bestG.get(key);
        if (known !== undefined && known <= ng) continue;
        const h = heuristic(next);
        if (h >= INF) continue;
        bestG.set(key, ng);
        const idx = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b); // after the push the player stands where the box was
        stParent.push(state);
        stFrom.push(b);
        stTo.push(dest);
        stDir.push(d);
        stG.push(ng);
        hPush({ f: fOf(ng, h), g: ng, seq: seq++, state: idx });
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The pushes, from the first to the last, then the moves: walk to each push's
  // standing square around the boxes as they then stand, then push.
  const chain = [];
  for (let s = found; s > 0; s = stParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = startPlayer;
  const now = startBoxes.slice();
  const occ = new Uint8Array(size);
  for (const s of chain) {
    const from = stFrom[s];
    const to = stTo[s];
    const d = stDir[s];
    const delta = DY[d] * w + DX[d];
    for (const b of now) occ[b] = 1;
    const walk = walkMoves(w, wall, occ, player, from - delta);
    for (const b of now) occ[b] = 0;
    if (walk === null) return null; // cannot happen: the search stood there itself
    moves += walk + PUSH[d];
    now[now.indexOf(from)] = to;
    player = from;
  }
  return { moves, pushes: chain.length, endPlayer: player };
}

/** One board solved by the fallback ladder: A*, then uniform cost, then greedy. */
function solveBoard(w, wall, goals, boxes, player, budgets) {
  for (const [mode, budget] of budgets) {
    const r = searchPushes(w, wall, goals, boxes, player, budget, mode);
    if (r !== null) return r;
  }
  return null;
}

/**
 * The level's rooms, or null when the level is not a clean set of rooms: cut the
 * dead squares of the player's row (the hallway — a box there reaches no goal),
 * flood-fill what is left, and take each part that holds boxes. Null when a box
 * starts in the hallway, a goal is cut away, or some part's boxes and goals differ
 * in number.
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const distTables = goals.map((g) => pushDistances(w, wall, g));
  const live = new Uint8Array(wall.length);
  for (let s = 0; s < wall.length; s++) {
    for (const dt of distTables) {
      if (dt[s] < INF) {
        live[s] = 1;
        break;
      }
    }
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  const stack = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const id = rooms.length;
    const squares = [s];
    part[s] = id;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop();
      for (let d = 0; d < 4; d++) {
        const q = p + DY[d] * w + DX[d];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = id;
          squares.push(q);
          stack.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  const kept = rooms.filter((r) => r.boxes.length > 0);
  if (kept.some((r) => r.boxes.length !== r.goals.length)) return null;
  kept.sort((a, b) => a.squares[0] - b.squares[0]);
  return kept;
}

/** A level's solution moves: room by room when it decomposes, whole otherwise. */
function solveLevel(level) {
  const { w, wall, goals, boxes, player } = level;
  if (boxes.length === 0) return '';
  if (player < 0 || boxes.length !== goals.length) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solveBoard(w, wall, goals, boxes, player, LEVEL_BUDGETS);
    return whole === null ? '' : whole.moves;
  }
  let moves = '';
  let at = player;
  for (const room of rooms) {
    const subWall = wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      // A finished room's boxes rest on its goals; a waiting room's stand where
      // they were drawn. Either way they are walls to this room's search.
      for (const b of other.done ? other.goals : other.boxes) subWall[b] = 1;
    }
    const solved = solveBoard(w, subWall, room.goals, room.boxes, at, ROOM_BUDGETS);
    if (solved === null) {
      // One room failed every fallback: try the whole level before giving up.
      const whole = solveBoard(w, wall, goals, boxes, player, LEVEL_BUDGETS);
      return whole === null ? '' : whole.moves;
    }
    moves += solved.moves;
    at = solved.endPlayer;
    room.done = true;
  }
  return moves;
}

/** The moves for a level in XSB: a string of u, d, l and r (an upper letter marks a push). */
function solve(text) {
  try {
    return solveLevel(parseLevel(text));
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  let levelFile;
  try {
    levelFile = join(I, id, 'level.txt');
    if (!statSync(levelFile).isFile()) continue;
  } catch {
    continue;
  }
  const moves = solve(readFileSync(levelFile, 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
