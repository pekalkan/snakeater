"""snakeater – continuous tube snake, finite shrinking CIRCLE, food + boost + shield (no audio)"""

import math
import random
import time
import pygame

# ---------------- Window / Pygame ----------------
pygame.init()
W, H = 1366, 768
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - shields added")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Menlo", 20)
TITLE_FONT = pygame.font.SysFont("Menlo", 48)

# ---------------- Colors ----------------
BG_COLOR   = (18, 22, 28)
SNAKE_BODY = (60, 190, 90)     # green body
SNAKE_HEAD = (110, 240, 140)   # lighter green head
SNAKE2_BODY = (90, 160, 220)   # player 2 body (blue)
SNAKE2_HEAD = (155, 205, 255)  # player 2 head (light blue)
FOOD_COLOR = (255, 255, 255)   # normal food = white
BOOST_COLOR = (255, 215, 0)    # golden boost food
HUD_BG     = (20, 20, 20, 140) # translucent HUD background
HUD_TEXT   = (235, 245, 235)
BTN_NORMAL  = (60, 190, 90)
BTN_HOVER   = (90, 210, 120)

BTN_TEXT    = (245, 255, 245)
SHIELD_COLOR   = (120, 255, 120)  # green shield orb

POISON_WARN_BG = (160, 30, 30, 160)
POISON_WARN_TEXT = (255, 230, 230)

# ---------------- Minimap (bottom-right overlay) ----------------
MINIMAP_SIZE   = 180        # square panel size (pixels)
MINIMAP_MARGIN = 14         # margin from screen edges
MINIMAP_BG     = (20, 20, 20, 170)
MINIMAP_BORDER = (120, 120, 120)

# ---------------- Finite World Bounds (CIRCULAR) ----------------
# Keep the original rectangle numbers only to infer an initial radius.
_RECT_W, _RECT_H = 4320, 2430
WORLD_DIAMETER_SCALE = 2.25                # +50% more (total 2.25x base diameter)
BASE_DIAMETER = min(_RECT_W, _RECT_H)
SAFE_R = int((BASE_DIAMETER * WORLD_DIAMETER_SCALE) / 2)   # scaled initial safe radius
SAFE_R_INIT = SAFE_R
OUTSIDE_DECAY_RATE = 180.0                 # length lost per second while outside

# --- Safe zone shrinking ---
SHRINK_INTERVAL = 60.0      # seconds between shrinks
SHRINK_FACTOR = 0.8         # each shrink scales world size by 80%
SHRINK_NOTICE_SECS = 3.0    # on-screen warning duration
MIN_WORLD_R = 800  # do not shrink below this radius (keeps area playable)
SHRINK_MESSAGE = "SHRINKING"
# runtime state for notices
shrink_notice_until = 0.0

# ---------------- Helpers ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def world_to_screen(px, py, camx, camy):
    return int(px - camx), int(py - camy)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def segment_intersection(p0, p1, p2, p3, eps: float = 1e-9):
    """
    Proper intersection test between two line segments p0->p1 and p2->p3.
    Returns (hit: bool, (ix, iy)).
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    x4, y4 = p3

    dx1, dy1 = (x2 - x1), (y2 - y1)
    dx2, dy2 = (x4 - x3), (y4 - y3)

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < eps:
        return False, (0.0, 0.0)  # parallel or nearly parallel

    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t * dx1
        iy = y1 + t * dy1
        return True, (ix, iy)
    return False, (0.0, 0.0)

# ---------------- Snake ----------------
class Snake:
    """Snake with length-limited trail; continuous tube rendering."""

    def __init__(self, x, y, controls, body_color, head_color):
        # Position / velocity
        self.x, self.y = float(x), float(y)
        self.vx = 0.0
        self.vy = 0.0

        # Movement model: cruise + boost-on-input + temporary golden boost
        self.base_speed = 120.0          # cruise speed (keys not pressed)
        self.heading_x, self.heading_y = 1.0, 0.0
        self.speed_current = self.base_speed

        # per-player controls and colors
        self.controls = controls            # (left, right, up, down)
        self.body_color = body_color
        self.head_color = head_color

        # Body / trail
        self.thickness = 12              # radius for caps / half-width of tube
        self.length = 250.0              # allowed trail length in pixels
        self.min_step = 2.0
        self.points = [(self.x, self.y)] # latest head at index 0

        # Temporary boost state
        self.boost_mul = 1.0
        self.boost_until = 0.0

        # Shield / poison state
        self.shield_charges = 0
        self.is_in_poison = False
        self.shield_until = 0.0          # NEW: timed shield

    # ---- movement & trail helpers ----
    def handle_input(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        left, right, up, down = self.controls
        dx = (1 if keys[right] else 0) - (1 if keys[left] else 0)
        dy = (1 if keys[down] else 0) - (1 if keys[up] else 0)
        mag = math.hypot(dx, dy)

        if mag:
            self.heading_x, self.heading_y = dx / mag, dy / mag

        # 2x while holding input, otherwise cruise at base speed
        intended = self.base_speed * (2.0 if mag else 1.0)
        # Size penalty: bigger snake moves slower (0.1% per food eaten; shields don't add length)
        size_penalty = 1.0 - max(0.0, (self.length - START_LENGTH)) * SPEED_DECAY_PER_LENGTH
        if size_penalty < MIN_SIZE_SPEED_FACTOR:
            size_penalty = MIN_SIZE_SPEED_FACTOR
        # Apply penalties and temporary boost
        self.speed_current = intended * size_penalty * self.boost_mul

        # Advance
        self.vx = self.heading_x * self.speed_current
        self.vy = self.heading_y * self.speed_current
        self.x += self.vx * dt
        self.y += self.vy * dt

    def _trim_trail_to_length(self) -> None:
        total = 0.0
        pts = self.points
        for i in range(len(pts) - 1):
            total += dist(pts[i], pts[i + 1])
        while total > self.length and len(pts) > 1:
            a, b = pts[-2], pts[-1]
            seg = dist(a, b)
            need = total - self.length
            if need >= seg:
                pts.pop()
                total -= seg
            else:
                t = (seg - need) / seg if seg else 0.0
                pts[-1] = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                break

    def _trail_length(self) -> float:
        s = 0.0
        for i in range(len(self.points) - 1):
            s += dist(self.points[i], self.points[i + 1])
        return s

    def cut_from_index(self, i: int, ip: tuple) -> float:
        """Cut the trail at segment index i with intersection point ip.
        Returns the length that was removed from the tail.
        Also updates self.length to the new actual trail length."""
        old_len = self._trail_length()
        # replace the vertex with the exact intersection, then truncate
        self.points[i] = ip
        self.points = self.points[: i + 1]
        new_len = self._trail_length()
        lost = max(0.0, old_len - new_len)
        self.length = new_len
        return lost

    def _self_cut_if_crossed(self) -> None:
        """Cut the snake if the newest head segment crosses an older segment."""
        if len(self.points) < 3:
            return
        p0 = self.points[0]
        p1 = self.points[1]
        skip = 6  # ignore very recent segments near head
        last_idx = len(self.points) - 1
        for i in range(1 + skip, last_idx):
            hit, ip = segment_intersection(p0, p1, self.points[i], self.points[i + 1])
            if hit:
                self.points[i] = ip
                self.points = self.points[: i + 1]
                self.length = self._trail_length()  # lose the cut piece permanently
                break

    def _push_head_samples(self, new_head):
        """Insert evenly spaced samples between previous head and new head."""
        prev = self.points[0]
        d = dist(new_head, prev)
        if d <= 1e-6:
            return
        desired = max(2.0, self.thickness * 0.8)
        steps = int(d / desired)
        steps = max(1, min(steps, 120))  # safety cap for performance
        step_x = (new_head[0] - prev[0]) / steps
        step_y = (new_head[1] - prev[1]) / steps
        for i in range(1, steps + 1):
            self.points.insert(0, (prev[0] + step_x * i, prev[1] + step_y * i))

    def grow(self, amount: float) -> None:
        self.length += float(amount)

    def apply_boost(self, mult: float, duration: float) -> None:
        now = time.time()
        if now < self.boost_until and self.boost_mul > 1.0:
            # Exponential stacking while already boosted
            self.boost_mul *= mult
            self.boost_until = now + duration
        else:
            self.boost_mul = mult
            self.boost_until = now + duration

    # ---- Shield helpers ----
    def is_shield_active(self) -> bool:
        return time.time() < self.shield_until

    def apply_shield(self, duration: float) -> None:
        now = time.time()
        if self.is_shield_active():
            self.shield_until += duration  # extend
        else:
            self.shield_until = now + duration

    # -------- respawn after being eaten --------
    def respawn(self, x: float, y: float) -> None:
        """Reset this snake at (x,y) with starting stats after being eaten."""
        self.x, self.y = float(x), float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.heading_x, self.heading_y = 1.0, 0.0
        self.speed_current = self.base_speed

        # Reset trail to a single point and restore starting length
        self.points = [(self.x, self.y)]
        self.length = START_LENGTH

        # Clear temporary states
        self.boost_mul = 1.0
        self.boost_until = 0.0
        self.shield_charges = 0
        self.is_in_poison = False
        self.shield_until = 0.0

    def update(self, dt: float) -> None:
        # Expire temporary boost
        if self.boost_mul != 1.0 and time.time() >= self.boost_until:
            self.boost_mul = 1.0

        self.handle_input(dt)
        self._push_head_samples((self.x, self.y))
        self._trim_trail_to_length()
        self._self_cut_if_crossed()

        # Poison zone handling (outside the world border)
        inside = (math.hypot(self.x, self.y) + self.thickness) <= SAFE_R
        self.is_in_poison = not inside
        if self.is_in_poison and not self.is_shield_active():
            old_len = self.length
            self.length = max(0.0, self.length - OUTSIDE_DECAY_RATE * dt)
            if self.length < old_len:
                self._trim_trail_to_length()

    def draw(self, surf: pygame.Surface, camx: float, camy: float) -> None:
        # Classic dot-trail rendering (smooth thanks to _push_head_samples)
        for p in self.points:
            sx, sy = world_to_screen(p[0], p[1], camx, camy)
            pygame.draw.circle(surf, self.body_color, (sx, sy), self.thickness)
        hx, hy = world_to_screen(self.x, self.y, camx, camy)

        # Shield aura (ring) if active
        if self.is_shield_active():
            pygame.draw.circle(surf, SHIELD_COLOR, (hx, hy), self.thickness + 6, 2)

        pygame.draw.circle(surf, self.head_color, (hx, hy), self.thickness + 2)

# ---------------- Infinite World (chunks) ----------------
CHUNK_SIZE = 800
FOOD_PER_CHUNK = 3
FOOD_R = 6
FOOD_GROWTH = 30.0
# Speed decay: each eaten food slows the snake by 10% (TEST)
SPEED_DECAY_PER_FOOD = 0.10
SPEED_DECAY_PER_LENGTH = SPEED_DECAY_PER_FOOD / FOOD_GROWTH  # per pixel of length gained
MIN_SIZE_SPEED_FACTOR = 0.5  # minimum cap: 50% of intended speed
BOOST_FRACTION = 0.05   # 5% of foods are boost orbs
BOOST_MULT     = 1.5    # move at 1.5x when boosted
BOOST_DURATION = 5.0    # boost lasts 5 seconds
PREDATION_RATIO = 1.0    # attacker must be > (ratio × defender.length) to steal

# --- Head-on predation: bigger snake can fully eat smaller on contact ---
EAT_ON_HEAD_COLLISION = True   # if True, head contact lets the larger snake eat the smaller
START_LENGTH = 250.0           # respawn starting trail length (matches Snake.__init__)

 # --- Shields (green protection) ---
SHIELD_FRACTION = 0.05  # 5% of foods are shields
SHIELD_DURATION = 5.0             # each pickup grants 5 s protection

# --- Defeat by length ---
LOSE_LENGTH = 60.0  # if a snake length <= 60, it loses

spawned_chunks = set()  # {(cx, cy)}
foods = []              # list of dicts: {"x","y","r","kind"}

def chunk_of(px: float, py: float):
    cx = math.floor(px / CHUNK_SIZE)
    cy = math.floor(py / CHUNK_SIZE)
    return int(cx), int(cy)

def chunk_intersects_world(cx: int, cy: int) -> bool:
    """Approximate: a chunk intersects the circular world if its rectangle gets within SAFE_R of the origin."""
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    right = left + CHUNK_SIZE
    bottom = top + CHUNK_SIZE
    # closest point on rect to (0,0)
    qx = clamp(0, left, right)
    qy = clamp(0, top, bottom)
    return math.hypot(qx, qy) <= SAFE_R

def random_spawn_inside_world(margin: int = 120):
    """Return a random (x,y) strictly inside the circular world, away from edge by 'margin'."""
    r = random.uniform(margin, max(margin, SAFE_R - margin))
    theta = random.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)

def shrink_world_centered(factor: float) -> None:
    global SAFE_R
    new_r = max(MIN_WORLD_R, int(SAFE_R * factor))
    if new_r == SAFE_R:
        return
    SAFE_R = new_r

def spawn_chunk(cx: int, cy: int) -> None:
    """Spawn items inside the part of the chunk that lies within the circular world."""
    global foods
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    margin = 20
    minx = left + margin
    maxx = left + CHUNK_SIZE - margin
    miny = top + margin
    maxy = top + CHUNK_SIZE - margin
    if minx >= maxx or miny >= maxy:
        return
    for _ in range(FOOD_PER_CHUNK):
        placed = False
        for __ in range(10):  # try up to 10 times to land inside the circle
            fx = random.uniform(minx, maxx)
            fy = random.uniform(miny, maxy)
            if math.hypot(fx, fy) <= SAFE_R - 10:
                r = random.random()
                if r < BOOST_FRACTION:
                    kind = "boost"
                elif r < BOOST_FRACTION + SHIELD_FRACTION:
                    kind = "shield"
                else:
                    kind = "normal"
                foods.append({"x": fx, "y": fy, "r": FOOD_R, "kind": kind})
                placed = True
                break
        if not placed:
            # if the chunk is almost entirely outside, skip silently
            pass

# ---------------- Spawn Balancer (per chunk) ----------------
TARGET_PER_CHUNK = FOOD_PER_CHUNK  # desired items per active chunk
MIN_BOOST_PER_CHUNK = 0            # no per-chunk boost guarantee
MIN_SHIELD_PER_CHUNK = 0           # no per-chunk shield guarantee (lets rate drop by ~50%)

def spawn_items_in_chunk(cx: int, cy: int, n: int, kind: str) -> None:
    if n <= 0:
        return
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    margin = 20
    minx = left + margin
    maxx = left + CHUNK_SIZE - margin
    miny = top + margin
    maxy = top + CHUNK_SIZE - margin
    if minx >= maxx or miny >= maxy:
        return
    for _ in range(n):
        for __ in range(10):
            fx = random.uniform(minx, maxx)
            fy = random.uniform(miny, maxy)
            if math.hypot(fx, fy) <= SAFE_R - 10:
                foods.append({"x": fx, "y": fy, "r": FOOD_R, "kind": kind})
                break


SPAWN_RADIUS_CHUNKS = 1
SPAWN_INTERVAL = 0.9
last_density_spawn = 0.0

# --- Proximity-based shield assurance (soft guarantee near each player) ---
NEAR_SHIELD_RADIUS = 900.0    # check radius around each player (pixels)
NEAR_SHIELD_TARGET = 1        # aim to have at least this many shields within radius
NEAR_SHIELD_COOLDOWN = 5.0    # at most one proximity spawn per this many seconds (global)
last_near_shield_spawn = 0.0

def _count_shields_near(px: float, py: float, radius: float) -> int:
    r2 = radius * radius
    c = 0
    for f in foods:
        if f.get("kind") == "shield":
            dx = f["x"] - px
            dy = f["y"] - py
            if (dx * dx + dy * dy) <= r2:
                c += 1
    return c

def _spawn_shield_near(px: float, py: float) -> bool:
    """Try to spawn one shield near (px,py) but still inside the safe circle."""
    # Try a few random polar offsets; keep within SAFE_R with a margin
    for _ in range(24):
        ang = random.uniform(0.0, 2.0 * math.pi)
        dist_min = 200.0
        dist_max = min(700.0, max(220.0, SAFE_R - 60.0))
        d = random.uniform(dist_min, dist_max)
        fx = px + math.cos(ang) * d
        fy = py + math.sin(ang) * d
        if math.hypot(fx, fy) <= SAFE_R - 10.0:
            foods.append({"x": fx, "y": fy, "r": FOOD_R, "kind": "shield"})
            return True
    return False

def maybe_spawn_nearby_shields(points: list[tuple[float, float]]) -> None:
    global last_near_shield_spawn
    now = time.time()
    if now - last_near_shield_spawn < NEAR_SHIELD_COOLDOWN:
        return
    # If any player lacks a nearby shield, spawn one near that player and cooldown
    for (px, py) in points:
        if _count_shields_near(px, py, NEAR_SHIELD_RADIUS) < NEAR_SHIELD_TARGET:
            if _spawn_shield_near(px, py):
                last_near_shield_spawn = now
                break

def periodic_spawn_around(points: list[tuple[float, float]]) -> None:
    """Every SPAWN_INTERVAL seconds, ensure each chunk near `points`
    reaches TARGET_PER_CHUNK total items and at least MIN_* per chunk boosts/shields."""
    global last_density_spawn
    now = time.time()
    if now - last_density_spawn < SPAWN_INTERVAL:
        return

    # Build the set of nearby chunk keys to maintain
    keys: set[tuple[int, int]] = set()
    for (px, py) in points:
        pcx, pcy = chunk_of(px, py)
        for dy in range(-SPAWN_RADIUS_CHUNKS, SPAWN_RADIUS_CHUNKS + 1):
            for dx in range(-SPAWN_RADIUS_CHUNKS, SPAWN_RADIUS_CHUNKS + 1):
                key = (pcx + dx, pcy + dy)
                if chunk_intersects_world(*key):
                    keys.add(key)

    if not keys:
        last_density_spawn = now
        return

    # Ensure chunks exist
    for key in keys:
        if key not in spawned_chunks:
            spawn_chunk(*key)
            spawned_chunks.add(key)

    # Count totals, boosts, shields per key
    total_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    boost_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    shield_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    for f in foods:
        k = chunk_of(f["x"], f["y"])
        if k in total_counts:
            total_counts[k] += 1
            fk = f.get("kind")
            if fk == "boost":
                boost_counts[k] += 1
            elif fk == "shield":
                shield_counts[k] += 1

    # Spawn deficits: guarantee at least 1 boost and 1 shield
    for key in keys:
        total = total_counts[key]
        boosts = boost_counts[key]
        shields = shield_counts[key]
        desired_total = TARGET_PER_CHUNK
        desired_boosts = max(MIN_BOOST_PER_CHUNK, int(round(desired_total * BOOST_FRACTION)))
        desired_shields = max(MIN_SHIELD_PER_CHUNK, int(round(desired_total * SHIELD_FRACTION)))
        missing_total = max(0, desired_total - total)

        missing_boosts = max(0, desired_boosts - boosts)
        if missing_boosts > 0:
            spawn_items_in_chunk(key[0], key[1], missing_boosts, "boost")
            missing_total -= missing_boosts

        missing_shields = max(0, desired_shields - shields)
        if missing_shields > 0:
            spawn_items_in_chunk(key[0], key[1], missing_shields, "shield")
            missing_total -= missing_shields

        if missing_total > 0:
            spawn_items_in_chunk(key[0], key[1], missing_total, "normal")

    last_density_spawn = now

def ensure_chunks_around(px: float, py: float, radius_chunks: int = 1) -> None:
    """Make sure the 3x3 neighborhood around the player is spawned."""
    pcx, pcy = chunk_of(px, py)
    for dy in range(-radius_chunks, radius_chunks + 1):
        for dx in range(-radius_chunks, radius_chunks + 1):
            key = (pcx + dx, pcy + dy)
            if key not in spawned_chunks and chunk_intersects_world(*key):
                spawn_chunk(*key)
                spawned_chunks.add(key)

def cull_far_foods(px: float, py: float, keep_radius_chunks: int = 2) -> None:
    """Remove foods too far away to keep memory bounded."""
    global foods
    pcx, pcy = chunk_of(px, py)
    keep = []
    for f in foods:
        cx, cy = chunk_of(f["x"], f["y"])
        if abs(cx - pcx) <= keep_radius_chunks and abs(cy - pcy) <= keep_radius_chunks:
            keep.append(f)
    foods = keep

def eat_food_if_colliding(snake: Snake) -> int:
    """Consume food that collides with the snake head. Returns number eaten."""
    global foods
    eaten = 0
    head = (snake.x, snake.y)
    keep = []
    head_r = snake.thickness
    for f in foods:
        if dist(head, (f["x"], f["y"])) <= (head_r + f["r"]):
            eaten += 1
            kind = f.get("kind")
            if kind == "shield":
                snake.apply_shield(SHIELD_DURATION)
            else:
                snake.grow(FOOD_GROWTH)
                if kind == "boost":
                    snake.apply_boost(BOOST_MULT, BOOST_DURATION)
        else:
            keep.append(f)
    foods = keep
    return eaten

def draw_foods(surf: pygame.Surface, camx: float, camy: float) -> None:
    for f in foods:
        sx, sy = world_to_screen(f["x"], f["y"], camx, camy)
        k = f.get("kind")
        if k == "boost":
            pygame.draw.circle(surf, BOOST_COLOR, (sx, sy), f["r"])  # gold filled
        elif k == "shield":
            # Bigger, pulsing, outlined to stand out clearly
            base_r = f["r"] + 3
            pulse = int(2 * (0.5 + 0.5 * math.sin(time.time() * 6)))  # 0..2
            pr = max(1, base_r + pulse)
            pygame.draw.circle(surf, SHIELD_COLOR, (sx, sy), pr)       # bright fill
            pygame.draw.circle(surf, (245, 255, 245), (sx, sy), pr, 2) # subtle outline
        else:
            pygame.draw.circle(surf, FOOD_COLOR, (sx, sy), f["r"])     # normal white

def draw_world_border(surf: pygame.Surface, camx: float, camy: float) -> None:
    cx, cy = world_to_screen(0, 0, camx, camy)
    pygame.draw.circle(surf, (80, 80, 80), (cx, cy), int(SAFE_R), 2)

# --- Poison Zone Blink Warning ---
def draw_poison_blink(surf: pygame.Surface) -> None:
    """Blinking red warning text shown when player is outside the world (poison zone)."""
    # Blink ~2Hz: visible one half-cycle, hidden the other half
    if (int(time.time() * 2) % 2) != 0:
        return
    msg = "WARNING, POISON!"
    txt = TITLE_FONT.render(msg, True, (255, 60, 60))
    shadow = TITLE_FONT.render(msg, True, (30, 0, 0))
    x = (surf.get_width() - txt.get_width()) // 2
    y = 18
    surf.blit(shadow, (x + 2, y + 2))
    surf.blit(txt, (x, y))


def draw_shrink_notice(surf: pygame.Surface) -> None:
    """Transient orange warning when the safe zone shrinks."""
    if time.time() >= shrink_notice_until:
        return
    msg = SHRINK_MESSAGE
    txt = TITLE_FONT.render(msg, True, (255, 200, 80))
    shadow = TITLE_FONT.render(msg, True, (25, 20, 0))
    x = (surf.get_width() - txt.get_width()) // 2
    y = 70
    surf.blit(shadow, (x + 2, y + 2))
    surf.blit(txt, (x, y))

# --- Minimap overlay (bottom-right) ---
def draw_minimap(surf: pygame.Surface, snakes: list[Snake]) -> None:
    """Render a simple circular minimap of the world with snake head dots in the
    bottom-right corner of the final screen buffer `surf`. World center is (0,0)."""
    # Panel placement
    panel_w = panel_h = MINIMAP_SIZE
    x0 = surf.get_width()  - panel_w - MINIMAP_MARGIN
    y0 = surf.get_height() - panel_h - MINIMAP_MARGIN

    # Panel surface with alpha background
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    pygame.draw.rect(panel, MINIMAP_BG, panel.get_rect(), border_radius=10)

    # Minimap center and radius
    cx = panel_w // 2
    cy = panel_h // 2
    # Fit the world circle into the panel with small padding
    pad = 10
    r_max = (min(panel_w, panel_h) // 2) - pad
    if SAFE_R <= 0:
        return
    scale = r_max / float(SAFE_R)

    # World border (scaled circle)
    pygame.draw.circle(panel, MINIMAP_BORDER, (cx, cy), int(SAFE_R * scale), 2)

    # Plot snake heads as small dots
    for i, s in enumerate(snakes):
        px = int(cx + s.x * scale)
        py = int(cy + s.y * scale)
        color = s.head_color  # use each snake's head color
        pygame.draw.circle(panel, color, (px, py), 4)

    # Blit panel to screen
    surf.blit(panel, (x0, y0))

# --- Overlay: Game Over / Win/Lose ---
def draw_game_over_overlay(surf: pygame.Surface, title: str) -> None:
    t = TITLE_FONT.render(title, True, (255, 230, 230))
    ts = TITLE_FONT.render(title, True, (40, 0, 0))
    x = (surf.get_width() - t.get_width()) // 2
    y = surf.get_height() // 2 - t.get_height()
    surf.blit(ts, (x + 2, y + 2))
    surf.blit(t,  (x, y))

def draw_start_menu(surf: pygame.Surface, btn1_rect: pygame.Rect, btn2_rect: pygame.Rect) -> None:
    surf.fill(BG_COLOR)
    # Title and hint
    title = TITLE_FONT.render("SNAKEATER", True, HUD_TEXT)
    hint  = FONT.render("Select 1 or 2 Players (press 1/2)", True, HUD_TEXT)
    surf.blit(title, ((W - title.get_width()) // 2, int(H * 0.32)))
    surf.blit(hint,  ((W - hint.get_width())  // 2, int(H * 0.32) + title.get_height() + 12))

    # Buttons (hover effect)
    mx, my = pygame.mouse.get_pos()

    for rect, label in ((btn1_rect, "1 PLAYER"), (btn2_rect, "2 PLAYERS")):
        hover = rect.collidepoint((mx, my))
        color = BTN_HOVER if hover else BTN_NORMAL
        pygame.draw.rect(surf, color, rect, border_radius=12)
        txt = FONT.render(label, True, BTN_TEXT)
        surf.blit(txt, (rect.x + (rect.width  - txt.get_width())  // 2,
                        rect.y + (rect.height - txt.get_height()) // 2))

# ---------------- Steal Mechanic ----------------
def steal_if_cross(attacker: Snake, defender: Snake, skip_recent: int = 4) -> float:
    """If the attacker's newest head segment crosses any older segment
    of the defender, cut the defender there and transfer the removed
    length to the attacker. Returns the stolen length (0.0 if none)."""
    if len(attacker.points) < 2 or len(defender.points) < 2:
        return 0.0
    # Defender shield blocks being cut
    if defender.is_shield_active():
        return 0.0
    # Predation rule: larger can eat smaller; smaller cannot eat larger
    if attacker.length <= PREDATION_RATIO * defender.length:
        return 0.0

    p0 = attacker.points[0]
    p1 = attacker.points[1]
    last_idx = len(defender.points) - 1

    # skip a few most-recent defender segments near its head
    for i in range(skip_recent, last_idx):
        hit, ip = segment_intersection(p0, p1, defender.points[i], defender.points[i + 1])
        if hit:
            stolen = defender.cut_from_index(i, ip)
            attacker.length += stolen
            return stolen
    return 0.0

# -------- Head-on predation helpers --------
def head_hits_snake(attacker: Snake, defender: Snake, skip_recent: int = 0) -> bool:
    """Return True if attacker's head circle overlaps any point of defender's tube."""
    head = (attacker.x, attacker.y)
    threshold = attacker.thickness + defender.thickness
    if len(defender.points) <= skip_recent:
        return False
    for p in defender.points[skip_recent:]:
        if dist(head, p) <= threshold:
            return True
    return False

def eat_if_head_collides(attacker: Snake, defender: Snake) -> bool:
    """
    If the larger attacker's head collides with the defender's tube/head,
    the attacker eats the defender completely (unless defender shielded).
    Transfers defender's total length to attacker and respawns defender.
    Returns True if an eat occurred (defender was respawned).
    """
    if not EAT_ON_HEAD_COLLISION:
        return False
    # Defender shield blocks being eaten
    if defender.is_shield_active():
        return False
    # Only larger snake can eat the smaller one
    if attacker.length <= PREDATION_RATIO * defender.length:
        return False
    if not head_hits_snake(attacker, defender, skip_recent=0):
        return False

    # Transfer all remaining length
    attacker.length += defender.length

    # Respawn defender at a new random location inside the world
    sx, sy = random_spawn_inside_world()
    defender.respawn(sx, sy)
    return True

# --- Elimination-style predation (no respawn) ---
def eat_and_maybe_eliminate(attacker: Snake, defender: Snake) -> str:
    """
    Returns: "none" (no eat) or "attacker_win" (defender eliminated; no respawn).
    """
    if not EAT_ON_HEAD_COLLISION:
        return "none"
    if defender.is_shield_active():
        return "none"
    if attacker.length <= PREDATION_RATIO * defender.length:
        return "none"
    if not head_hits_snake(attacker, defender, skip_recent=0):
        return "none"
    # direct elimination (no respawn): attacker absorbs and round ends
    attacker.length += defender.length
    return "attacker_win"

# ---------------- HUD ----------------
def draw_hud_one(surf: pygame.Surface, s: Snake, eaten: int, label: str = "") -> None:
    sh_left = max(0, int(s.shield_until - time.time()))
    sh_text = f"   SHD {sh_left:2d}s" if sh_left > 0 else ""
    text = f"{label} LEN {int(s.length):4d}   SPD {int(s.speed_current):3d}   FOOD {eaten:3d}{sh_text}".strip()
    txt = FONT.render(text, True, HUD_TEXT)
    pad = 10
    w = txt.get_width() + pad * 2
    h = txt.get_height() + pad * 2
    hud = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(hud, HUD_BG, hud.get_rect(), border_radius=10)
    hud.blit(txt, (pad, pad))
    surf.blit(hud, (12, 10))

def draw_hud_two(surf: pygame.Surface, s1: Snake, eaten1: int, s2: Snake, eaten2: int) -> None:
    # left panel (P1)
    sh1 = max(0, int(s1.shield_until - time.time()))
    sh2 = max(0, int(s2.shield_until - time.time()))
    text1 = f"P1 LEN {int(s1.length):4d}   SPD {int(s1.speed_current):3d}   FOOD {eaten1:3d}" + (f"   SHD {sh1:2d}s" if sh1 > 0 else "")
    txt1 = FONT.render(text1, True, HUD_TEXT)
    pad = 10
    w1 = txt1.get_width() + pad * 2
    h1 = txt1.get_height() + pad * 2
    hud1 = pygame.Surface((w1, h1), pygame.SRCALPHA)
    pygame.draw.rect(hud1, HUD_BG, hud1.get_rect(), border_radius=10)
    hud1.blit(txt1, (pad, pad))
    surf.blit(hud1, (12, 10))

    # right panel (P2)
    text2 = f"P2 LEN {int(s2.length):4d}   SPD {int(s2.speed_current):3d}   FOOD {eaten2:3d}" + (f"   SHD {sh2:2d}s" if sh2 > 0 else "")
    txt2 = FONT.render(text2, True, HUD_TEXT)
    w2 = txt2.get_width() + pad * 2
    h2 = txt2.get_height() + pad * 2
    hud2 = pygame.Surface((w2, h2), pygame.SRCALPHA)
    pygame.draw.rect(hud2, HUD_BG, hud2.get_rect(), border_radius=10)
    hud2.blit(txt2, (pad, pad))
    surf.blit(hud2, (W - w2 - 12, 10))

# ---------------- Main Loop ----------------
controls_p1 = (pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)              # left, right, up, down
controls_p2 = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)   # left, right, up, down

snake1 = Snake(-120.0, 0.0, controls_p1, SNAKE_BODY, SNAKE_HEAD)
snake2 = Snake( 120.0, 0.0, controls_p2, SNAKE2_BODY, SNAKE2_HEAD)

total_eaten1 = 0
total_eaten2 = 0

def main():
    global total_eaten1, total_eaten2, SAFE_R, shrink_notice_until, snake1, snake2
    running = True
    game_state = "menu"
    game_mode = None  # "1p" or "2p"
    game_over = False
    loser = None

    # Safe zone shrink scheduling
    next_shrink_time = time.time() + SHRINK_INTERVAL

    # Two buttons centered with a small gap
    btn_w, btn_h, gap = 260, 64, 40
    left_x = (W - (btn_w * 2 + gap)) // 2
    btn1_rect = pygame.Rect(left_x, int(H * 0.55), btn_w, btn_h)
    btn2_rect = pygame.Rect(left_x + btn_w + gap, int(H * 0.55), btn_w, btn_h)

    while running:
        dt = clock.tick(60) / 1000.0

        # Periodic safe zone shrink
        now = time.time()
        if now >= next_shrink_time:
            shrink_world_centered(SHRINK_FACTOR)
            shrink_notice_until = now + SHRINK_NOTICE_SECS
            next_shrink_time = now + SHRINK_INTERVAL

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                # Restart: reset world, foods, snakes, timers, flags
                foods.clear()
                spawned_chunks.clear()
                total_eaten1 = 0
                total_eaten2 = 0
                # reset world radius
                SAFE_R = SAFE_R_INIT
                # reset snakes
                snake1 = Snake(-120.0, 0.0, controls_p1, SNAKE_BODY, SNAKE_HEAD)
                snake2 = Snake( 120.0, 0.0, controls_p2, SNAKE2_BODY, SNAKE2_HEAD)
                # timers & flags
                next_shrink_time = time.time() + SHRINK_INTERVAL
                shrink_notice_until = 0.0
                game_over = False
                loser = None
                # preserve current mode (stay in game if selected)
                if game_mode in ("1p", "2p"):
                    game_state = "game"
                else:
                    game_state = "menu"
                continue

            if game_state == "menu":
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_1:
                        game_mode = "1p"; game_state = "game"
                    elif e.key == pygame.K_2:
                        game_mode = "2p"; game_state = "game"
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if btn1_rect.collidepoint(e.pos):
                        game_mode = "1p"; game_state = "game"
                    elif btn2_rect.collidepoint(e.pos):
                        game_mode = "2p"; game_state = "game"

        if game_state == "menu":
            draw_start_menu(screen, btn1_rect, btn2_rect)
            pygame.display.flip()
            continue

        # --- Game state ---
        if game_mode == "1p":
            # World management for P1 only
            ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
            cull_far_foods(snake1.x, snake1.y, keep_radius_chunks=2)
            periodic_spawn_around([(snake1.x, snake1.y)])

            # Update P1
            snake1.update(dt)
            total_eaten1 += eat_food_if_colliding(snake1)

            # Camera follows P1 (full-screen)
            camx = snake1.x - W / 2
            camy = snake1.y - H / 2

            # Draw full-screen
            screen.fill(BG_COLOR)
            draw_world_border(screen, camx, camy)
            draw_foods(screen, camx, camy)
            snake1.draw(screen, camx, camy)
            if snake1.is_in_poison: draw_poison_blink(screen)
            draw_shrink_notice(screen)
            draw_hud_one(screen, snake1, total_eaten1, "P1")
            draw_minimap(screen, [snake1])
            pygame.display.flip()
            continue

        # --- 2P split-screen ---
        # Spawn around both; cull around midpoint
        ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
        ensure_chunks_around(snake2.x, snake2.y, radius_chunks=1)
        midx = 0.5 * (snake1.x + snake2.x)
        midy = 0.5 * (snake1.y + snake2.y)
        cull_far_foods(midx, midy, keep_radius_chunks=2)
        periodic_spawn_around([(snake1.x, snake1.y), (snake2.x, snake2.y)])

        # Update both
        snake1.update(dt)
        snake2.update(dt)
        total_eaten1 += eat_food_if_colliding(snake1)
        total_eaten2 += eat_food_if_colliding(snake2)

        # Head-on predation (elimination logic)
        if not game_over:
            res = eat_and_maybe_eliminate(snake1, snake2)
            if res == "attacker_win":
                game_over = True
                loser = "P2"
            else:
                res2 = eat_and_maybe_eliminate(snake2, snake1)
                if res2 == "attacker_win":
                    game_over = True
                    loser = "P1"

        # Steal mechanic both ways – shield blocks being cut
        _ = steal_if_cross(snake1, snake2)
        _ = steal_if_cross(snake2, snake1)

        # Length-based defeat
        if not game_over:
            if snake1.length <= LOSE_LENGTH:
                game_over = True
                loser = "P1"
            elif snake2.length <= LOSE_LENGTH:
                game_over = True
                loser = "P2"

        # Cameras per player
        view_w = W // 2
        cam1x = snake1.x - view_w / 2
        cam1y = snake1.y - H / 2
        cam2x = snake2.x - view_w / 2
        cam2y = snake2.y - H / 2

        # Render to two view surfaces
        left = pygame.Surface((view_w, H))
        right = pygame.Surface((view_w, H))

        # Left view (P1)
        left.fill(BG_COLOR)
        draw_world_border(left, cam1x, cam1y)
        draw_foods(left, cam1x, cam1y)
        snake1.draw(left, cam1x, cam1y)
        snake2.draw(left, cam1x, cam1y)
        if snake1.is_in_poison: draw_poison_blink(left)
        draw_shrink_notice(left)
        draw_hud_one(left, snake1, total_eaten1, "P1")

        # Right view (P2)
        right.fill(BG_COLOR)
        draw_world_border(right, cam2x, cam2y)
        draw_foods(right, cam2x, cam2y)
        snake1.draw(right, cam2x, cam2y)
        snake2.draw(right, cam2x, cam2y)
        if snake2.is_in_poison: draw_poison_blink(right)
        draw_shrink_notice(right)
        draw_hud_one(right, snake2, total_eaten2, "P2")

        # Draw overlay for win/lose
        if game_over and loser == "P1":
            draw_game_over_overlay(left, "YOU LOSE")
            draw_game_over_overlay(right, "YOU WIN")
        elif game_over and loser == "P2":
            draw_game_over_overlay(left, "YOU WIN")
            draw_game_over_overlay(right, "YOU LOSE")

        # Compose to screen
        screen.blit(left, (0, 0))
        screen.blit(right, (view_w, 0))
        pygame.draw.rect(screen, (30, 30, 30), pygame.Rect(view_w - 1, 0, 2, H))
        draw_minimap(screen, [snake1, snake2])
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()