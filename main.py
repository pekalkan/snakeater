"""snakeater – continuous tube snake, finite shrinking CIRCLE, food + boost + shield + AI BOTS"""

import math
import random
import time
import pygame
import os

# ---------------- Window / Pygame ----------------
pygame.init()
pygame.mixer.init()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_PATH = os.path.join(BASE_DIR, "assets", "background.ogg")
MUSIC_VOLUME = 0.5

# ---- SFX (shield, mine, poison, nitro, net) ----
SFX_DIR = os.path.join(BASE_DIR, "assets")
SFX_SHIELD_PATH  = os.path.join(SFX_DIR, "shield.mp3")
SFX_MINE_PATH    = os.path.join(SFX_DIR, "mine.mp3")
SFX_POISON_PATH  = os.path.join(SFX_DIR, "poison.mp3")
SFX_NITRO_PATH   = os.path.join(SFX_DIR, "boost.mp3")
SFX_NET_PATH     = os.path.join(SFX_DIR, "net.mp3")
SFX_SHRINK_PATH  = os.path.join(SFX_DIR, "shrinking.mp3")
SFX_VOLUME = 0.7

# Load sounds
try:
    sfx_shield = pygame.mixer.Sound(SFX_SHIELD_PATH)
    sfx_shield.set_volume(SFX_VOLUME)
except Exception as _e:
    sfx_shield = None
    print(f"[sfx] shield not loaded: {SFX_SHIELD_PATH} ({_e})")
try:
    sfx_mine = pygame.mixer.Sound(SFX_MINE_PATH)
    sfx_mine.set_volume(SFX_VOLUME)
except Exception as _e:
    sfx_mine = None
    print(f"[sfx] mine not loaded: {SFX_MINE_PATH} ({_e})")
try:
    sfx_poison = pygame.mixer.Sound(SFX_POISON_PATH)
    sfx_poison.set_volume(0.5)
except Exception as _e:
    sfx_poison = None
    print(f"[sfx] poison not loaded: {SFX_POISON_PATH} ({_e})")
try:
    sfx_nitro = pygame.mixer.Sound(SFX_NITRO_PATH)
    sfx_nitro.set_volume(SFX_VOLUME)
except Exception as _e:
    sfx_nitro = None
    print(f"[sfx] nitro not loaded: {SFX_NITRO_PATH} ({_e})")
try:
    sfx_net = pygame.mixer.Sound(SFX_NET_PATH)
    sfx_net.set_volume(SFX_VOLUME)
except Exception as _e:
    sfx_net = None
    print(f"[sfx] net not loaded: {SFX_NET_PATH} ({_e})")

try:
    sfx_shrink = pygame.mixer.Sound(SFX_SHRINK_PATH)
    sfx_shrink.set_volume(SFX_VOLUME)
except Exception as _e:
    sfx_shrink = None
    print(f"[sfx] shrink not loaded: {SFX_SHRINK_PATH} ({_e})")

# Dedicated channel for looping poison ambience
POISON_CH_INDEX = 5
poison_ch = pygame.mixer.Channel(POISON_CH_INDEX)
poison_looping = False

def set_poison_loop(should_play: bool):
    global poison_looping
    if should_play and not poison_looping and sfx_poison is not None:
        poison_ch.play(sfx_poison, loops=-1, fade_ms=150)
        poison_looping = True
    elif not should_play and poison_looping:
        poison_ch.fadeout(200)
        poison_looping = False

W, H = 1366, 768
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - Single Player AI Added")
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

 # HUD extra colors
SPEED_YELLOW   = (255, 220, 0)
CD_READY       = (120, 255, 120)
CD_WAIT        = (255, 180, 70)

# ---------------- Music helpers ----------------
def play_bg_music(path: str = MUSIC_PATH, volume: float = MUSIC_VOLUME) -> None:
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1, fade_ms=800)
    except Exception as e:
        print(f"[music] Could not start background music: {e}")

# ---------------- Minimap (bottom-right overlay) ----------------
MINIMAP_SIZE   = 180
MINIMAP_MARGIN = 14
MINIMAP_BG     = (20, 20, 20, 170)
MINIMAP_BORDER = (120, 120, 120)

# ---------------- Finite World Bounds (CIRCULAR) ----------------
_RECT_W, _RECT_H = 4320, 2430
WORLD_DIAMETER_SCALE = 6.0
BASE_DIAMETER = min(_RECT_W, _RECT_H)
SAFE_R = int((BASE_DIAMETER * WORLD_DIAMETER_SCALE) / 2)
SAFE_R = int(SAFE_R * math.sqrt(2.0))
SAFE_R_INIT = SAFE_R
OUTSIDE_DECAY_RATE = 180.0

# --- Safe zone shrinking ---
SHRINK_INTERVAL = 60.0
SHRINK_FACTOR = 0.8
SHRINK_NOTICE_SECS = 3.0
MIN_WORLD_R = 800
SHRINK_MESSAGE = "SHRINKING"
shrink_notice_until = 0.0

# ---------------- Helpers ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def world_to_screen(px, py, camx, camy):
    return int(px - camx), int(py - camy)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def segment_intersection(p0, p1, p2, p3, eps: float = 1e-9):
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    x4, y4 = p3

    dx1, dy1 = (x2 - x1), (y2 - y1)
    dx2, dy2 = (x4 - x3), (y4 - y3)

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < eps:
        return False, (0.0, 0.0)

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
        self.x, self.y = float(x), float(y)
        self.vx = 0.0
        self.vy = 0.0

        self.base_speed = 120.0
        self.heading_x, self.heading_y = 1.0, 0.0
        self.speed_current = self.base_speed

        self.controls = controls # (left, right, up, down)
        self.body_color = body_color
        self.head_color = head_color

        self.thickness = 12
        self.length = 250.0
        self.min_step = 2.0
        self.points = [(self.x, self.y)]

        self.boost_mul = 1.0
        self.boost_until = 0.0

        self.shield_charges = 0
        self.is_in_poison = False
        self.shield_until = 0.0

    def handle_input(self, dt: float) -> None:
        # Standard Keyboard Input
        if not self.controls or self.controls[0] is None:
            return # For Bots, we override this method

        keys = pygame.key.get_pressed()
        left, right, up, down = self.controls
        dx = (1 if keys[right] else 0) - (1 if keys[left] else 0)
        dy = (1 if keys[down] else 0) - (1 if keys[up] else 0)
        mag = math.hypot(dx, dy)

        if mag:
            self.heading_x, self.heading_y = dx / mag, dy / mag

        intended = self.base_speed * (2.0 if mag else 1.0)
        size_penalty = 1.0 - max(0.0, (self.length - START_LENGTH)) * SPEED_DECAY_PER_LENGTH
        if size_penalty < MIN_SIZE_SPEED_FACTOR:
            size_penalty = MIN_SIZE_SPEED_FACTOR
        self.speed_current = intended * size_penalty * self.boost_mul

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
        old_len = self._trail_length()
        self.points[i] = ip
        self.points = self.points[: i + 1]
        new_len = self._trail_length()
        lost = max(0.0, old_len - new_len)
        self.length = new_len
        return lost

    def _self_cut_if_crossed(self) -> None:
        if len(self.points) < 3:
            return
        p0 = self.points[0]
        p1 = self.points[1]
        skip = 6
        last_idx = len(self.points) - 1
        for i in range(1 + skip, last_idx):
            hit, ip = segment_intersection(p0, p1, self.points[i], self.points[i + 1])
            if hit:
                self.points[i] = ip
                self.points = self.points[: i + 1]
                self.length = self._trail_length()
                break

    def _push_head_samples(self, new_head):
        prev = self.points[0]
        d = dist(new_head, prev)
        if d <= 1e-6:
            return
        desired = max(2.0, self.thickness * 0.8)
        steps = int(d / desired)
        steps = max(1, min(steps, 120))
        step_x = (new_head[0] - prev[0]) / steps
        step_y = (new_head[1] - prev[1]) / steps
        for i in range(1, steps + 1):
            self.points.insert(0, (prev[0] + step_x * i, prev[1] + step_y * i))

    def grow(self, amount: float) -> None:
        self.length += float(amount)

    def apply_boost(self, mult: float, duration: float) -> None:
        now = time.time()
        if now < self.boost_until and self.boost_mul > 1.0:
            self.boost_mul *= mult
            self.boost_until = now + duration
        else:
            self.boost_mul = mult
            self.boost_until = now + duration

    def is_shield_active(self) -> bool:
        return time.time() < self.shield_until

    def apply_shield(self, duration: float) -> None:
        now = time.time()
        if self.is_shield_active():
            self.shield_until += duration
        else:
            self.shield_until = now + duration

    def respawn(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.heading_x, self.heading_y = 1.0, 0.0
        self.speed_current = self.base_speed
        self.points = [(self.x, self.y)]
        self.length = START_LENGTH
        self.boost_mul = 1.0
        self.boost_until = 0.0
        self.shield_charges = 0
        self.is_in_poison = False
        self.shield_until = 0.0

    def update(self, dt: float) -> None:
        if self.boost_mul != 1.0 and time.time() >= self.boost_until:
            self.boost_mul = 1.0

        self.handle_input(dt)
        self._push_head_samples((self.x, self.y))
        self._trim_trail_to_length()
        self._self_cut_if_crossed()

        inside = (math.hypot(self.x, self.y) + self.thickness) <= SAFE_R
        self.is_in_poison = not inside
        if self.is_in_poison and not self.is_shield_active():
            old_len = self.length
            self.length = max(0.0, self.length - OUTSIDE_DECAY_RATE * dt)
            if self.length < old_len:
                self._trim_trail_to_length()

    def draw(self, surf: pygame.Surface, camx: float, camy: float) -> None:
        for p in self.points:
            sx, sy = world_to_screen(p[0], p[1], camx, camy)
            pygame.draw.circle(surf, self.body_color, (sx, sy), self.thickness)
        hx, hy = world_to_screen(self.x, self.y, camx, camy)

        if self.is_shield_active():
            pygame.draw.circle(surf, SHIELD_COLOR, (hx, hy), self.thickness + 6, 2)

        pygame.draw.circle(surf, self.head_color, (hx, hy), self.thickness + 2)


# ---------------- Bot AI Class ----------------
class BotSnake(Snake):
    """
    AI-controlled snake that inherits from the main Snake class.
    It automatically seeks food, avoids the poison zone, and wanders.
    """
    def __init__(self, x, y, body_color, head_color):
        # Pass None for controls since bots don't use keyboard
        super().__init__(x, y, (None, None, None, None), body_color, head_color)
        self.change_dir_timer = 0.0
        self.target_x = 0
        self.target_y = 0
        # Turn speed determines how fast the bot adjusts its angle (steering)
        self.turn_speed = 5.0 

    def handle_input(self, dt: float) -> None:
        """AI decision making instead of keyboard input."""
        
        # 1. Safety Check (Highest Priority): Stay inside SAFE_R
        dist_to_center = math.hypot(self.x, self.y)
        # If getting close to the edge (margin of 150px), turn back to center
        if dist_to_center > SAFE_R - 150:
            desired_vx = -self.x
            desired_vy = -self.y
        else:
            # 2. Food Hunting (Medium Priority)
            # Find the closest food visible
            closest_food = None
            min_dist = 1000.0  # Perception radius
            
            # Access global foods list
            for f in foods:
                d = math.hypot(f["x"] - self.x, f["y"] - self.y)
                if d < min_dist:
                    # Avoid mines intentionally
                    if f.get("kind") == "mine":
                        continue
                    min_dist = d
                    closest_food = f

            if closest_food:
                desired_vx = closest_food["x"] - self.x
                desired_vy = closest_food["y"] - self.y
                
                # Randomly use boost if target is far away
                if min_dist > 250 and random.random() < 0.015:
                    self.apply_boost(BOOST_MULT, 1.0)
            else:
                # 3. Wandering (Low Priority)
                # Pick a random direction every few seconds
                self.change_dir_timer -= dt
                if self.change_dir_timer <= 0:
                    angle = random.uniform(0, 2 * math.pi)
                    self.target_x = math.cos(angle)
                    self.target_y = math.sin(angle)
                    self.change_dir_timer = random.uniform(1.0, 3.0)
                
                desired_vx = self.target_x
                desired_vy = self.target_y

        # Steering Behavior: Smoothly rotate heading towards desired velocity
        curr_ang = math.atan2(self.heading_y, self.heading_x)
        target_ang = math.atan2(desired_vy, desired_vx)
        
        # Calculate shortest rotation logic
        diff = target_ang - curr_ang
        while diff <= -math.pi: diff += 2*math.pi
        while diff > math.pi: diff -= 2*math.pi
        
        # Apply turn speed
        turn_amount = self.turn_speed * dt
        if abs(diff) < turn_amount:
            curr_ang = target_ang
        else:
            curr_ang += turn_amount if diff > 0 else -turn_amount
            
        self.heading_x = math.cos(curr_ang)
        self.heading_y = math.sin(curr_ang)

        # Set speed
        self.speed_current = self.base_speed * self.boost_mul
        
        # Apply movement
        self.vx = self.heading_x * self.speed_current
        self.vy = self.heading_y * self.speed_current
        self.x += self.vx * dt
        self.y += self.vy * dt


# ---------------- Infinite World (chunks) ----------------
CHUNK_SIZE = 800
FOOD_PER_CHUNK = 3
FOOD_R = 6
FOOD_R_MIN = 4
FOOD_R_MAX = 10
FOOD_GROWTH = 30.0
SPEED_DECAY_PER_FOOD = 0.01
SPEED_DECAY_PER_LENGTH = SPEED_DECAY_PER_FOOD / FOOD_GROWTH
MIN_SIZE_SPEED_FACTOR = 0.5
BOOST_FRACTION = 0.04
BOOST_MULT     = 1.5
BOOST_DURATION = 5.0
PREDATION_RATIO = 1.0

EAT_ON_HEAD_COLLISION = True
START_LENGTH = 250.0

SHIELD_FRACTION = 0.05
SHIELD_DURATION = 5.0

MINE_FRACTION = 0.05
MINE_ARM_TIME = 3.0
MINE_BLAST_RADIUS = 180.0
MINE_COLOR = (220, 60, 60)
MINE_RING  = (255, 140, 140)

NET_COOLDOWN = 20.0
NET_OFFSET_PIX = 220.0
NET_DURATION = 2.0
NET_RADIUS_PER_LEN = 0.25
NET_RADIUS_MIN = 120.0
NET_RADIUS_MAX = 420.0
NET_DECAY_RATE = 360.0
NET_IGNORES_SHIELD = False
NET_COLOR = (220, 80, 220, 90)
NET_RING = (235, 180, 245)

LOSE_LENGTH = 60.0

spawned_chunks = set()
foods = []
nets = []
mines = []

def chunk_of(px: float, py: float):
    cx = math.floor(px / CHUNK_SIZE)
    cy = math.floor(py / CHUNK_SIZE)
    return int(cx), int(cy)

def chunk_intersects_world(cx: int, cy: int) -> bool:
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    right = left + CHUNK_SIZE
    bottom = top + CHUNK_SIZE
    qx = clamp(0, left, right)
    qy = clamp(0, top, bottom)
    return math.hypot(qx, qy) <= SAFE_R

def random_spawn_inside_world(margin: int = 120):
    r = random.uniform(margin, max(margin, SAFE_R - margin))
    theta = random.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)

def _make_food(x: float, y: float, kind: str) -> dict:
    r = random.uniform(FOOD_R_MIN, FOOD_R_MAX)
    r_i = int(round(r))
    if kind in ("shield", "mine"):
        growth = 0.0
    else:
        growth = FOOD_GROWTH * (r / float(FOOD_R))
    return {"x": x, "y": y, "r": r_i, "kind": kind, "growth": growth}

def shrink_world_centered(factor: float) -> None:
    global SAFE_R
    new_r = max(MIN_WORLD_R, int(SAFE_R * factor))
    if new_r == SAFE_R:
        return
    SAFE_R = new_r

def spawn_chunk(cx: int, cy: int) -> None:
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
        for __ in range(10):
            fx = random.uniform(minx, maxx)
            fy = random.uniform(miny, maxy)
            if math.hypot(fx, fy) <= SAFE_R - 10:
                r = random.random()
                if r < BOOST_FRACTION:
                    kind = "boost"
                elif r < BOOST_FRACTION + SHIELD_FRACTION:
                    kind = "shield"
                elif r < BOOST_FRACTION + SHIELD_FRACTION + MINE_FRACTION:
                    kind = "mine"
                else:
                    kind = "normal"
                foods.append(_make_food(fx, fy, kind))
                placed = True
                break
        if not placed:
            pass

# ---------------- Spawn Balancer ----------------
TARGET_PER_CHUNK = FOOD_PER_CHUNK
MIN_BOOST_PER_CHUNK = 0
MIN_SHIELD_PER_CHUNK = 0
MIN_MINE_PER_CHUNK = 0

def spawn_mixed_in_chunk(cx: int, cy: int, n: int) -> None:
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
                r = random.random()
                if r < BOOST_FRACTION:
                    kind = "boost"
                elif r < BOOST_FRACTION + SHIELD_FRACTION:
                    kind = "shield"
                elif r < BOOST_FRACTION + SHIELD_FRACTION + MINE_FRACTION:
                    kind = "mine"
                else:
                    kind = "normal"
                foods.append(_make_food(fx, fy, kind))
                break

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
                foods.append(_make_food(fx, fy, kind))
                break

SPAWN_RADIUS_CHUNKS = 1
SPAWN_INTERVAL = 0.9
last_density_spawn = 0.0

NEAR_SHIELD_RADIUS = 900.0
NEAR_SHIELD_TARGET = 1
NEAR_SHIELD_COOLDOWN = 5.0
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
    for _ in range(24):
        ang = random.uniform(0.0, 2.0 * math.pi)
        dist_min = 200.0
        dist_max = min(700.0, max(220.0, SAFE_R - 60.0))
        d = random.uniform(dist_min, dist_max)
        fx = px + math.cos(ang) * d
        fy = py + math.sin(ang) * d
        if math.hypot(fx, fy) <= SAFE_R - 10.0:
            foods.append(_make_food(fx, fy, "shield"))
            return True
    return False

def maybe_spawn_nearby_shields(points: list[tuple[float, float]]) -> None:
    global last_near_shield_spawn
    now = time.time()
    if now - last_near_shield_spawn < NEAR_SHIELD_COOLDOWN:
        return
    for (px, py) in points:
        if _count_shields_near(px, py, NEAR_SHIELD_RADIUS) < NEAR_SHIELD_TARGET:
            if _spawn_shield_near(px, py):
                last_near_shield_spawn = now
                break

def periodic_spawn_around(points: list[tuple[float, float]]) -> None:
    global last_density_spawn
    now = time.time()
    if now - last_density_spawn < SPAWN_INTERVAL:
        return

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

    for key in keys:
        if key not in spawned_chunks:
            spawn_chunk(*key)
            spawned_chunks.add(key)

    total_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    boost_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    shield_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    mine_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    for f in foods:
        k = chunk_of(f["x"], f["y"])
        if k in total_counts:
            total_counts[k] += 1
            fk = f.get("kind")
            if fk == "boost":
                boost_counts[k] += 1
            elif fk == "shield":
                shield_counts[k] += 1
            elif fk == "mine":
                mine_counts[k] += 1

    for key in keys:
        total   = total_counts[key]
        boosts  = boost_counts[key]
        shields = shield_counts[key]
        mines_c = mine_counts[key]

        desired_total   = TARGET_PER_CHUNK
        desired_boosts  = max(MIN_BOOST_PER_CHUNK,  int(round(desired_total * BOOST_FRACTION)))
        desired_shields = max(MIN_SHIELD_PER_CHUNK, int(round(desired_total * SHIELD_FRACTION)))
        desired_mines   = max(MIN_MINE_PER_CHUNK,   int(round(desired_total * MINE_FRACTION)))

        missing_total = max(0, desired_total - total)

        missing_boosts = max(0, desired_boosts - boosts)
        if missing_boosts > 0:
            spawn_items_in_chunk(key[0], key[1], missing_boosts, "boost")
            missing_total -= missing_boosts

        missing_shields = max(0, desired_shields - shields)
        if missing_shields > 0:
            spawn_items_in_chunk(key[0], key[1], missing_shields, "shield")
            missing_total -= missing_shields

        missing_mines = max(0, desired_mines - mines_c)
        if missing_mines > 0:
            spawn_items_in_chunk(key[0], key[1], missing_mines, "mine")
            missing_total -= missing_mines

        if missing_total > 0:
            spawn_mixed_in_chunk(key[0], key[1], missing_total)

    last_density_spawn = now

def ensure_chunks_around(px: float, py: float, radius_chunks: int = 1) -> None:
    pcx, pcy = chunk_of(px, py)
    for dy in range(-radius_chunks, radius_chunks + 1):
        for dx in range(-radius_chunks, radius_chunks + 1):
            key = (pcx + dx, pcy + dy)
            if key not in spawned_chunks and chunk_intersects_world(*key):
                spawn_chunk(*key)
                spawned_chunks.add(key)

# --- Net helpers ---
def _net_radius_for(s: Snake) -> float:
    r = NET_RADIUS_PER_LEN * max(0.0, s.length)
    return clamp(r, NET_RADIUS_MIN, NET_RADIUS_MAX)

def cast_net(caster: Snake) -> None:
    now = time.time()
    if hasattr(caster, "last_net_time") and now - caster.last_net_time < NET_COOLDOWN:
        return
    caster.last_net_time = now

    off = NET_OFFSET_PIX
    nx = caster.x + caster.heading_x * off
    ny = caster.y + caster.heading_y * off
    nets.append({
        "x": nx,
        "y": ny,
        "r": _net_radius_for(caster),
        "owner": caster,
        "until": now + NET_DURATION,
    })
    if 'sfx_net' in globals() and sfx_net:
        sfx_net.play()

def update_nets() -> None:
    now = time.time()
    keep = [n for n in nets if now < n["until"]]
    nets[:] = keep

def apply_net_effect(snakes: list[Snake], dt: float) -> None:
    if not nets:
        return
    for s in snakes:
        harmful = False
        for n in nets:
            if n["owner"] is s:
                continue
            dx = s.x - n["x"]
            dy = s.y - n["y"]
            if (dx*dx + dy*dy) <= (n["r"] * n["r"]):
                harmful = True
                break
        if harmful:
            if NET_IGNORES_SHIELD or not s.is_shield_active():
                old_len = s.length
                s.length = max(0.0, s.length - NET_DECAY_RATE * dt)
                if s.length < old_len:
                    s._trim_trail_to_length()

def cull_far_foods(px: float, py: float, keep_radius_chunks: int = 2) -> None:
    global foods
    pcx, pcy = chunk_of(px, py)
    keep = []
    for f in foods:
        cx, cy = chunk_of(f["x"], f["y"])
        if abs(cx - pcx) <= keep_radius_chunks and abs(cy - pcy) <= keep_radius_chunks:
            keep.append(f)
    foods = keep

def eat_food_if_colliding(snake: Snake) -> int:
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
                if sfx_shield: 
                    sfx_shield.play()
            elif kind == "mine":
                mines.append({
                    "x": f["x"],
                    "y": f["y"],
                    "blast_r": float(MINE_BLAST_RADIUS),
                    "detonate_at": time.time() + MINE_ARM_TIME,
                })
            else:
                growth = f.get("growth", FOOD_GROWTH)
                snake.grow(growth)
                if kind == "boost":
                    snake.apply_boost(BOOST_MULT, BOOST_DURATION)
                    if 'sfx_nitro' in globals() and sfx_nitro:
                        sfx_nitro.play()
        else:
            keep.append(f)
    foods = keep
    return eaten

def draw_foods(surf: pygame.Surface, camx: float, camy: float) -> None:
    for f in foods:
        sx, sy = world_to_screen(f["x"], f["y"], camx, camy)
        k = f.get("kind")
        if k == "boost":
            base_r = f["r"] + 2
            pulse = int(2 * (0.5 + 0.5 * math.sin(time.time() * 6)))
            pr = max(1, base_r + pulse)
            pygame.draw.circle(surf, BOOST_COLOR, (sx, sy), pr)
            pygame.draw.circle(surf, (255, 240, 130), (sx, sy), pr, 2)
        elif k == "shield":
            base_r = f["r"] + 3
            pulse = int(2 * (0.5 + 0.5 * math.sin(time.time() * 6)))
            pr = max(1, base_r + pulse)
            pygame.draw.circle(surf, SHIELD_COLOR, (sx, sy), pr)
            pygame.draw.circle(surf, (245, 255, 245), (sx, sy), pr, 2)
        elif k == "mine":
            base_r = f["r"] + 3
            pulse = int(2 * (0.5 + 0.5 * math.sin(time.time() * 5)))
            pr = max(1, base_r + pulse)
            pygame.draw.circle(surf, MINE_COLOR, (sx, sy), pr)
            pygame.draw.circle(surf, (255, 220, 220), (sx, sy), pr, 2)
        else:
            pygame.draw.circle(surf, FOOD_COLOR, (sx, sy), f["r"])

def draw_world_border(surf: pygame.Surface, camx: float, camy: float) -> None:
    cx, cy = world_to_screen(0, 0, camx, camy)
    pygame.draw.circle(surf, (80, 80, 80), (cx, cy), int(SAFE_R), 2)

def draw_nets(surf: pygame.Surface, camx: float, camy: float) -> None:
    if not nets:
        return
    for n in nets:
        sx, sy = world_to_screen(n["x"], n["y"], camx, camy)
        disk = pygame.Surface((int(n["r"]*2)+4, int(n["r"]*2)+4), pygame.SRCALPHA)
        pygame.draw.circle(disk, NET_COLOR, (disk.get_width()//2, disk.get_height()//2), int(n["r"]))
        pygame.draw.circle(disk, NET_RING, (disk.get_width()//2, disk.get_height()//2), int(n["r"]), 2)
        surf.blit(disk, (sx - disk.get_width()//2, sy - disk.get_height()//2))

def draw_mines(surf: pygame.Surface, camx: float, camy: float) -> None:
    if not mines:
        return
    now = time.time()
    for m in mines:
        sx, sy = world_to_screen(m["x"], m["y"], camx, camy)
        r = int(m["blast_r"])
        arm_left = max(0.0, m["detonate_at"] - now)
        disk = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(disk, (MINE_COLOR[0], MINE_COLOR[1], MINE_COLOR[2], 60), (disk.get_width()//2, disk.get_height()//2), r)
        pygame.draw.circle(disk, MINE_RING, (disk.get_width()//2, disk.get_height()//2), r, 3)
        surf.blit(disk, (sx - disk.get_width()//2, sy - disk.get_height()//2))
        ct = int(math.ceil(arm_left))
        if ct > 0:
            t = FONT.render(str(ct), True, (255, 230, 230))
            surf.blit(t, (sx - t.get_width()//2, sy - t.get_height()//2))

def _any_body_point_in_circle(s: Snake, cx: float, cy: float, r: float) -> bool:
    r2 = r * r
    for p in s.points:
        dx = p[0] - cx
        dy = p[1] - cy
        if (dx*dx + dy*dy) <= r2:
            return True
    return False

def update_and_detonate_mines(snakes: list[Snake]) -> None:
    if not mines:
        return
    now = time.time()
    keep = []
    for m in mines:
        if now < m["detonate_at"]:
            keep.append(m)
            continue
        if sfx_mine:
            sfx_mine.play()
        cx, cy, br = m["x"], m["y"], m["blast_r"]
        for s in snakes:
            if math.hypot(s.x - cx, s.y - cy) <= br:
                s.length = 0.0
                s._trim_trail_to_length()
                continue
            if _any_body_point_in_circle(s, cx, cy, br):
                s.length = max(0.0, s.length * 0.5)
                s._trim_trail_to_length()
    mines[:] = keep

def draw_poison_blink(surf: pygame.Surface) -> None:
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
    if time.time() >= shrink_notice_until:
        return
    msg = SHRINK_MESSAGE
    txt = TITLE_FONT.render(msg, True, (255, 200, 80))
    shadow = TITLE_FONT.render(msg, True, (25, 20, 0))
    x = (surf.get_width() - txt.get_width()) // 2
    y = 70
    surf.blit(shadow, (x + 2, y + 2))
    surf.blit(txt, (x, y))

def draw_minimap(surf: pygame.Surface, snakes: list[Snake]) -> None:
    panel_w = panel_h = MINIMAP_SIZE
    x0 = surf.get_width()  - panel_w - MINIMAP_MARGIN
    y0 = surf.get_height() - panel_h - MINIMAP_MARGIN

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    pygame.draw.rect(panel, MINIMAP_BG, panel.get_rect(), border_radius=10)

    cx = panel_w // 2
    cy = panel_h // 2
    pad = 10
    r_max = (min(panel_w, panel_h) // 2) - pad
    if SAFE_R <= 0:
        return
    scale = r_max / float(SAFE_R)

    pygame.draw.circle(panel, MINIMAP_BORDER, (cx, cy), int(SAFE_R * scale), 2)

    for i, s in enumerate(snakes):
        px = int(cx + s.x * scale)
        py = int(cy + s.y * scale)
        color = s.head_color
        pygame.draw.circle(panel, color, (px, py), 4)

    if snakes:
        spd = int(snakes[0].speed_current)
        spd_txt = FONT.render(f"SPD {spd}", True, HUD_TEXT)
        panel.blit(spd_txt, (8, panel_h - spd_txt.get_height() - 6))

    surf.blit(panel, (x0, y0))

def draw_game_over_overlay(surf: pygame.Surface, title: str) -> None:
    t = TITLE_FONT.render(title, True, (255, 60, 60))
    ts = TITLE_FONT.render(title, True, (120, 0, 0))
    x = (surf.get_width() - t.get_width()) // 2
    y = surf.get_height() // 2 - t.get_height()
    surf.blit(ts, (x + 2, y + 2))
    surf.blit(t,  (x, y))

def draw_restart_hint(surf: pygame.Surface) -> None:
    hint = FONT.render("Press R to restart", True, (235, 245, 235))
    x = (surf.get_width() - hint.get_width()) // 2
    y = surf.get_height() // 2 + 12
    surf.blit(hint, (x, y))

def _render_hud_segments(segments: list[tuple[str, tuple[int, int, int]]]) -> pygame.Surface:
    pieces = [FONT.render(txt, True, col) for (txt, col) in segments if txt]
    if not pieces:
        return pygame.Surface((1, 1), pygame.SRCALPHA)
    w = sum(p.get_width() for p in pieces)
    h = max(p.get_height() for p in pieces)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    x = 0
    for p in pieces:
        surf.blit(p, (x, 0))
        x += p.get_width()
    return surf

def draw_start_menu(surf: pygame.Surface, btn1_rect: pygame.Rect, btn2_rect: pygame.Rect) -> None:
    surf.fill(BG_COLOR)
    title = TITLE_FONT.render("SNAKEATER", True, HUD_TEXT)
    hint  = FONT.render("Select 1 or 2 Players (press 1/2)", True, HUD_TEXT)
    surf.blit(title, ((W - title.get_width()) // 2, int(H * 0.32)))
    surf.blit(hint,  ((W - hint.get_width())  // 2, int(H * 0.32) + title.get_height() + 12))

    mx, my = pygame.mouse.get_pos()
    for rect, label in ((btn1_rect, "1 PLAYER (vs AI)"), (btn2_rect, "2 PLAYERS")):
        hover = rect.collidepoint((mx, my))
        color = BTN_HOVER if hover else BTN_NORMAL
        pygame.draw.rect(surf, color, rect, border_radius=12)
        txt = FONT.render(label, True, BTN_TEXT)
        surf.blit(txt, (rect.x + (rect.width  - txt.get_width())  // 2,
                        rect.y + (rect.height - txt.get_height()) // 2))

def steal_if_cross(attacker: Snake, defender: Snake, skip_recent: int = 4) -> float:
    if len(attacker.points) < 2 or len(defender.points) < 2:
        return 0.0
    if defender.is_shield_active():
        return 0.0
    if attacker.length <= PREDATION_RATIO * defender.length:
        return 0.0

    p0 = attacker.points[0]
    p1 = attacker.points[1]
    last_idx = len(defender.points) - 1

    for i in range(skip_recent, last_idx):
        hit, ip = segment_intersection(p0, p1, defender.points[i], defender.points[i + 1])
        if hit:
            stolen = defender.cut_from_index(i, ip)
            attacker.length += stolen
            return stolen
    return 0.0

def head_hits_snake(attacker: Snake, defender: Snake, skip_recent: int = 0) -> bool:
    head = (attacker.x, attacker.y)
    threshold = attacker.thickness + defender.thickness
    if len(defender.points) <= skip_recent:
        return False
    for p in defender.points[skip_recent:]:
        if dist(head, p) <= threshold:
            return True
    return False

def eat_if_head_collides(attacker: Snake, defender: Snake) -> bool:
    if not EAT_ON_HEAD_COLLISION:
        return False
    if defender.is_shield_active():
        return False
    if attacker.length <= PREDATION_RATIO * defender.length:
        return False
    if not head_hits_snake(attacker, defender, skip_recent=0):
        return False

    attacker.length += defender.length
    sx, sy = random_spawn_inside_world()
    defender.respawn(sx, sy)
    return True

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
    attacker.length += defender.length
    return "attacker_win"

def draw_hud_one(surf: pygame.Surface, s: Snake, eaten: int, label: str = "") -> None:
    now = time.time()
    sh_left = max(0, int(s.shield_until - now))
    if hasattr(s, "last_net_time"):
        cd = max(0, int(NET_COOLDOWN - (now - s.last_net_time)))
    else:
        cd = 0

    segs: list[tuple[str, tuple[int, int, int]]] = []
    if label:
        segs.append((f"{label} ", HUD_TEXT))
    segs.extend([
        ("LEN ", HUD_TEXT), (f"{int(s.length):4d}", HUD_TEXT), ("   ", HUD_TEXT),
        ("SPD ", HUD_TEXT), (f"{int(s.speed_current):3d}", SPEED_YELLOW), ("   ", HUD_TEXT),
        ("FOOD ", HUD_TEXT), (f"{eaten:3d}", HUD_TEXT),
    ])
    if sh_left > 0:
        segs.extend([("   SHD ", CD_READY), (f"{sh_left:2d}s", CD_READY)])
    if cd == 0:
        segs.extend([("   NET ", HUD_TEXT), ("READY", CD_READY)])
    else:
        segs.extend([("   NET ", HUD_TEXT), (f"{cd:2d}s", CD_WAIT)])

    txt = _render_hud_segments(segs)
    pad = 10
    w = txt.get_width() + pad * 2
    h = txt.get_height() + pad * 2
    hud = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(hud, HUD_BG, hud.get_rect(), border_radius=10)
    hud.blit(txt, (pad, pad))
    surf.blit(hud, (12, 10))

def draw_hud_two(surf: pygame.Surface, s1: Snake, eaten1: int, s2: Snake, eaten2: int) -> None:
    now = time.time()
    pad = 10

    # ---- P1 panel ----
    sh1 = max(0, int(s1.shield_until - now))
    cd1 = max(0, int(NET_COOLDOWN - (now - getattr(s1, "last_net_time", 0)))) if hasattr(s1, "last_net_time") else 0
    segs1: list[tuple[str, tuple[int, int, int]]] = [
        ("P1 ", HUD_TEXT),
        ("LEN ", HUD_TEXT), (f"{int(s1.length):4d}", HUD_TEXT), ("   ", HUD_TEXT),
        ("SPD ", HUD_TEXT), (f"{int(s1.speed_current):3d}", SPEED_YELLOW), ("   ", HUD_TEXT),
        ("FOOD ", HUD_TEXT), (f"{eaten1:3d}", HUD_TEXT),
    ]
    if sh1 > 0:
        segs1.extend([("   SHD ", CD_READY), (f"{sh1:2d}s", CD_READY)])
    if cd1 == 0:
        segs1.extend([("   NET ", HUD_TEXT), ("READY", CD_READY)])
    else:
        segs1.extend([("   NET ", HUD_TEXT), (f"{cd1:2d}s", CD_WAIT)])

    txt1 = _render_hud_segments(segs1)
    w1 = txt1.get_width() + pad * 2
    h1 = txt1.get_height() + pad * 2
    hud1 = pygame.Surface((w1, h1), pygame.SRCALPHA)
    pygame.draw.rect(hud1, HUD_BG, hud1.get_rect(), border_radius=10)
    hud1.blit(txt1, (pad, pad))
    surf.blit(hud1, (12, 10))

    # ---- P2 panel ----
    sh2 = max(0, int(s2.shield_until - now))
    cd2 = max(0, int(NET_COOLDOWN - (now - getattr(s2, "last_net_time", 0)))) if hasattr(s2, "last_net_time") else 0
    segs2: list[tuple[str, tuple[int, int, int]]] = [
        ("P2 ", HUD_TEXT),
        ("LEN ", HUD_TEXT), (f"{int(s2.length):4d}", HUD_TEXT), ("   ", HUD_TEXT),
        ("SPD ", HUD_TEXT), (f"{int(s2.speed_current):3d}", SPEED_YELLOW), ("   ", HUD_TEXT),
        ("FOOD ", HUD_TEXT), (f"{eaten2:3d}", HUD_TEXT),
    ]
    if sh2 > 0:
        segs2.extend([("   SHD ", CD_READY), (f"{sh2:2d}s", CD_READY)])
    if cd2 == 0:
        segs2.extend([("   NET ", HUD_TEXT), ("READY", CD_READY)])
    else:
        segs2.extend([("   NET ", HUD_TEXT), (f"{cd2:2d}s", CD_WAIT)])

    txt2 = _render_hud_segments(segs2)
    w2 = txt2.get_width() + pad * 2
    h2 = txt2.get_height() + pad * 2
    hud2 = pygame.Surface((w2, h2), pygame.SRCALPHA)
    pygame.draw.rect(hud2, HUD_BG, hud2.get_rect(), border_radius=10)
    hud2.blit(txt2, (pad, pad))
    surf.blit(hud2, (W - w2 - 12, 10))

def place_players_opposite_edges() -> None:
    global snake1, snake2
    ang = random.uniform(0.0, 2.0 * math.pi)
    margin = 160.0
    r = max(120.0, SAFE_R - margin)
    x1 = math.cos(ang) * r
    y1 = math.sin(ang) * r
    x2 = -x1
    y2 = -y1
    snake1.respawn(x1, y1)
    snake2.respawn(x2, y2)
    def _face_in(s: Snake):
        dd = math.hypot(-s.x, -s.y)
        if dd > 1e-6:
            s.heading_x = (-s.x) / dd
            s.heading_y = (-s.y) / dd
    _face_in(snake1)
    _face_in(snake2)

def place_players_random() -> None:
    global snake1, snake2
    x1, y1 = random_spawn_inside_world()
    x2, y2 = random_spawn_inside_world()
    snake1.respawn(x1, y1)
    snake2.respawn(x2, y2)
    for s in (snake1, snake2):
        ang = random.uniform(0.0, 2.0 * math.pi)
        s.heading_x = math.cos(ang)
        s.heading_y = math.sin(ang)

# ---------------- Main Loop ----------------
controls_p1 = (pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)
controls_p2 = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)

snake1 = Snake(-120.0, 0.0, controls_p1, SNAKE_BODY, SNAKE_HEAD)
snake2 = Snake( 120.0, 0.0, controls_p2, SNAKE2_BODY, SNAKE2_HEAD)

total_eaten1 = 0
total_eaten2 = 0

def main():
    global total_eaten1, total_eaten2, SAFE_R, shrink_notice_until, snake1, snake2, last_density_spawn, last_near_shield_spawn
    
    # --- BOT SETUP ---
    bots = []
    BOT_COUNT = 6
    
    def spawn_bots():
        """Helper to clear and respawn bots with random colors."""
        bots.clear()
        for i in range(BOT_COUNT):
            bx, by = random_spawn_inside_world()
            # Generate random distinct colors for bots
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            b_head = (r, g, b)
            b_body = (max(0, r-40), max(0, g-40), max(0, b-40))
            
            new_bot = BotSnake(bx, by, b_body, b_head)
            bots.append(new_bot)
            
    running = True
    game_state = "menu"
    game_mode = None  # "1p" or "2p"
    game_over = False
    loser = None
    music_paused = False
    play_bg_music()
    prev_p1_poison = False
    prev_p2_poison = False

    next_shrink_time = time.time() + SHRINK_INTERVAL

    btn_w, btn_h, gap = 260, 64, 40
    left_x = (W - (btn_w * 2 + gap)) // 2
    btn1_rect = pygame.Rect(left_x, int(H * 0.55), btn_w, btn_h)
    btn2_rect = pygame.Rect(left_x + btn_w + gap, int(H * 0.55), btn_w, btn_h)

    while running:
        dt = clock.tick(60) / 1000.0

        now = time.time()
        if game_state == "game" and now >= next_shrink_time:
            shrink_world_centered(SHRINK_FACTOR)
            shrink_notice_until = now + SHRINK_NOTICE_SECS
            next_shrink_time = now + SHRINK_INTERVAL
            if 'sfx_shrink' in globals() and sfx_shrink:
                sfx_shrink.play(maxtime=3000)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                game_state = "menu"
                game_mode = None
                game_over = False
                loser = None
                shrink_notice_until = 0.0
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                # Restart
                foods.clear()
                spawned_chunks.clear()
                nets.clear()
                mines.clear()
                last_density_spawn = 0.0
                last_near_shield_spawn = 0.0
                total_eaten1 = 0
                total_eaten2 = 0
                SAFE_R = SAFE_R_INIT
                snake1 = Snake(-120.0, 0.0, controls_p1, SNAKE_BODY, SNAKE_HEAD)
                snake2 = Snake( 120.0, 0.0, controls_p2, SNAKE2_BODY, SNAKE2_HEAD)
                
                spawn_bots() # Respawn bots on reset
                
                if game_mode == "2p":
                    place_players_random()
                next_shrink_time = time.time() + SHRINK_INTERVAL
                shrink_notice_until = 0.0
                game_over = False
                loser = None
                if game_mode in ("1p", "2p"):
                    game_state = "game"
                else:
                    game_state = "menu"
                continue

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_e:
                    cast_net(snake1)
                elif e.key == pygame.K_RSHIFT:
                    cast_net(snake2)
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                    if music_paused:
                        pygame.mixer.music.unpause()
                        music_paused = False
                    else:
                        pygame.mixer.music.pause()
                        music_paused = True

            if game_state == "menu":
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_1:
                        game_mode = "1p"; game_state = "game"
                        spawn_bots() # Spawn bots for single player
                    elif e.key == pygame.K_2:
                        game_mode = "2p"; game_state = "game"
                        place_players_random()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if btn1_rect.collidepoint(e.pos):
                        game_mode = "1p"; game_state = "game"
                        spawn_bots() # Spawn bots for single player
                    elif btn2_rect.collidepoint(e.pos):
                        game_mode = "2p"; game_state = "game"
                        place_players_random()

        if game_state == "menu":
            draw_start_menu(screen, btn1_rect, btn2_rect)
            pygame.display.flip()
            continue

        # --- Game state ---
        if game_mode == "1p":
            # 1. World management
            ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
            cull_far_foods(snake1.x, snake1.y, keep_radius_chunks=2)
            periodic_spawn_around([(snake1.x, snake1.y)])
            update_nets()

            # 2. Update Player 1
            snake1.update(dt)
            total_eaten1 += eat_food_if_colliding(snake1)
            apply_net_effect([snake1] + bots, dt) # Apply nets to player AND bots
            
            # 3. Update Bots
            for b in bots:
                b.update(dt)
                eat_food_if_colliding(b) # Bots eat food too
                
                # Bot vs Player interactions
                # Player tries to eat Bot
                res = eat_and_maybe_eliminate(snake1, b)
                if res == "attacker_win":
                    # Respawn the eaten bot elsewhere to keep game lively
                    bx, by = random_spawn_inside_world()
                    b.respawn(bx, by)
                
                # Bot tries to eat Player
                res_p = eat_and_maybe_eliminate(b, snake1)
                if res_p == "attacker_win":
                    # Player loses
                    game_over = True
                    loser = "P1"
                
                # Steal mechanics (cutting trails)
                steal_if_cross(snake1, b)
                steal_if_cross(b, snake1)

            # Update mines for everyone
            update_and_detonate_mines([snake1] + bots)

            # Poison ambience
            set_poison_loop(snake1.is_in_poison)

            # Camera follows P1
            camx = snake1.x - W / 2
            camy = snake1.y - H / 2

            # Draw
            screen.fill(BG_COLOR)
            draw_world_border(screen, camx, camy)
            draw_nets(screen, camx, camy)
            draw_mines(screen, camx, camy)
            draw_foods(screen, camx, camy)
            
            # Draw Bots
            for b in bots:
                b.draw(screen, camx, camy)
                
            snake1.draw(screen, camx, camy)
            
            if snake1.is_in_poison: draw_poison_blink(screen)
            draw_shrink_notice(screen)
            draw_hud_one(screen, snake1, total_eaten1, "P1")
            draw_mode_badge(screen, "MODE: 1P vs AI")
            
            # Show bots on minimap alongside player
            draw_minimap(screen, [snake1] + bots)
            
            # Game Over Overlay
            if game_over:
                draw_game_over_overlay(screen, "GAME OVER")
                draw_restart_hint(screen)

            pygame.display.flip()
            continue

        # --- 2P split-screen ---
        ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
        ensure_chunks_around(snake2.x, snake2.y, radius_chunks=1)
        midx = 0.5 * (snake1.x + snake2.x)
        midy = 0.5 * (snake1.y + snake2.y)
        cull_far_foods(midx, midy, keep_radius_chunks=4)
        periodic_spawn_around([(snake1.x, snake1.y), (snake2.x, snake2.y)])
        update_nets()

        if not game_over:
            snake1.update(dt)
            snake2.update(dt)
            total_eaten1 += eat_food_if_colliding(snake1)
            total_eaten2 += eat_food_if_colliding(snake2)
            apply_net_effect([snake1, snake2], dt)
            update_and_detonate_mines([snake1, snake2])

            set_poison_loop(snake1.is_in_poison or snake2.is_in_poison)
            prev_p1_poison = snake1.is_in_poison
            prev_p2_poison = snake2.is_in_poison

            res = eat_and_maybe_eliminate(snake1, snake2)
            if res == "attacker_win":
                game_over = True
                loser = "P2"
            else:
                res2 = eat_and_maybe_eliminate(snake2, snake1)
                if res2 == "attacker_win":
                    game_over = True
                    loser = "P1"

            if not game_over:
                _ = steal_if_cross(snake1, snake2)
                _ = steal_if_cross(snake2, snake1)

            if not game_over:
                if snake1.length <= LOSE_LENGTH:
                    game_over = True
                    loser = "P1"
                elif snake2.length <= LOSE_LENGTH:
                    game_over = True
                    loser = "P2"

        view_w = W // 2
        cam1x = snake1.x - view_w / 2
        cam1y = snake1.y - H / 2
        cam2x = snake2.x - view_w / 2
        cam2y = snake2.y - H / 2

        left = pygame.Surface((view_w, H))
        right = pygame.Surface((view_w, H))

        # Left view (P1)
        left.fill(BG_COLOR)
        draw_world_border(left, cam1x, cam1y)
        draw_nets(left, cam1x, cam1y)
        draw_mines(left, cam1x, cam1y)
        draw_foods(left, cam1x, cam1y)
        snake1.draw(left, cam1x, cam1y)
        snake2.draw(left, cam1x, cam1y)
        if snake1.is_in_poison: draw_poison_blink(left)
        draw_shrink_notice(left)
        draw_hud_one(left, snake1, total_eaten1, "P1")
        draw_mode_badge(left, "MODE: 2P")
        draw_minimap(left, [snake1])

        # Right view (P2)
        right.fill(BG_COLOR)
        draw_world_border(right, cam2x, cam2y)
        draw_nets(right, cam2x, cam2y)
        draw_mines(right, cam2x, cam2y)
        draw_foods(right, cam2x, cam2y)
        snake1.draw(right, cam2x, cam2y)
        snake2.draw(right, cam2x, cam2y)
        if snake2.is_in_poison: draw_poison_blink(right)
        draw_shrink_notice(right)
        draw_hud_one(right, snake2, total_eaten2, "P2")
        draw_mode_badge(right, "MODE: 2P")
        draw_minimap(right, [snake2])

        if game_over and loser == "P1":
            draw_game_over_overlay(left, "YOU LOSE")
            draw_restart_hint(left)
            draw_game_over_overlay(right, "YOU WIN")
            draw_restart_hint(right)
        elif game_over and loser == "P2":
            draw_game_over_overlay(left, "YOU WIN")
            draw_restart_hint(left)
            draw_game_over_overlay(right, "YOU LOSE")
            draw_restart_hint(right)

        screen.blit(left, (0, 0))
        screen.blit(right, (view_w, 0))
        pygame.draw.rect(screen, (30, 30, 30), pygame.Rect(view_w - 1, 0, 2, H))
        pygame.display.flip()

    pygame.quit()

def draw_mode_badge(surf: pygame.Surface, text: str) -> None:
    badge = FONT.render(text, True, HUD_TEXT)
    pad = 8
    w = badge.get_width() + 2 * pad
    h = badge.get_height() + 2 * pad
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(box, HUD_BG, box.get_rect(), border_radius=8)
    box.blit(badge, (pad, pad))
    surf.blit(box, (12, 50))


if __name__ == "__main__":
    main()