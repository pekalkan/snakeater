"""snakeater – continuous tube snake, infinite world, food + boost"""

import math
import random
import time
import pygame

# ---------------- Window / Pygame ----------------
pygame.init()
W, H = 1366, 768
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - step 2 (infinite world + food)")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Menlo", 20)
TITLE_FONT = pygame.font.SysFont("Menlo", 48)

# ---------------- Colors ----------------
BG_COLOR   = (18, 22, 28)
SNAKE_BODY = (60, 190, 90)     # green body
SNAKE_HEAD = (110, 240, 140)   # lighter green head
SNAKE2_BODY = (90, 160, 220)    # player 2 body (blue)
SNAKE2_HEAD = (155, 205, 255)   # player 2 head (light blue)
FOOD_COLOR = (255, 255, 255)   # normal food = white
BOOST_COLOR = (255, 215, 0)    # golden boost food
HUD_BG     = (20, 20, 20, 140) # translucent HUD background
HUD_TEXT   = (235, 245, 235)
BTN_NORMAL  = (60, 190, 90)
BTN_HOVER   = (90, 210, 120)

BTN_TEXT    = (245, 255, 245)

# ---------------- Finite World Bounds ----------------
WORLD_W, WORLD_H = 3000, 2000        # reasonable finite world size
WORLD_LEFT  = -WORLD_W // 2
WORLD_TOP   = -WORLD_H // 2
WORLD_RIGHT = WORLD_LEFT + WORLD_W
WORLD_BOTTOM= WORLD_TOP + WORLD_H

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
        # Apply temporary boost multiplier (if any)
        self.speed_current = intended * self.boost_mul

        # Advance
        self.vx = self.heading_x * self.speed_current
        self.vy = self.heading_y * self.speed_current
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Constrain to finite world bounds
        self.x = clamp(self.x, WORLD_LEFT + self.thickness, WORLD_RIGHT - self.thickness)
        self.y = clamp(self.y, WORLD_TOP  + self.thickness, WORLD_BOTTOM - self.thickness)

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

    def update(self, dt: float) -> None:
        # Expire temporary boost
        if self.boost_mul != 1.0 and time.time() >= self.boost_until:
            self.boost_mul = 1.0

        self.handle_input(dt)
        self._push_head_samples((self.x, self.y))
        self._trim_trail_to_length()
        self._self_cut_if_crossed()

    def draw(self, surf: pygame.Surface, camx: float, camy: float) -> None:
        # Classic dot-trail rendering (smooth thanks to _push_head_samples)
        for p in self.points:
            sx, sy = world_to_screen(p[0], p[1], camx, camy)
            pygame.draw.circle(surf, self.body_color, (sx, sy), self.thickness)
        hx, hy = world_to_screen(self.x, self.y, camx, camy)
        pygame.draw.circle(surf, self.head_color, (hx, hy), self.thickness + 2)

# ---------------- Infinite World (chunks) ----------------
CHUNK_SIZE = 800
FOOD_PER_CHUNK = 12
FOOD_R = 6
FOOD_GROWTH = 30.0

BOOST_FRACTION = 0.03   # 3% of foods are boost orbs
BOOST_MULT     = 1.5    # move at 1.5x when boosted
BOOST_DURATION = 5.0    # boost lasts 5 seconds
PREDATION_RATIO = 1.0    # attacker must be > (ratio × defender.length) to steal

spawned_chunks = set()  # {(cx, cy)}
foods = []              # list of dicts: {"x","y","r","kind"}

def chunk_of(px: float, py: float):
    cx = math.floor(px / CHUNK_SIZE)
    cy = math.floor(py / CHUNK_SIZE)
    return int(cx), int(cy)


def chunk_intersects_world(cx: int, cy: int) -> bool:
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    right = left + CHUNK_SIZE
    bottom = top + CHUNK_SIZE
    return not (right <= WORLD_LEFT or left >= WORLD_RIGHT or bottom <= WORLD_TOP or top >= WORLD_BOTTOM)

def spawn_chunk(cx: int, cy: int) -> None:
    """Create FOOD_PER_CHUNK food items randomly within the intersection of
    the chunk and the finite world. Each food has BOOST_FRACTION chance to be boost."""
    global foods
    left = cx * CHUNK_SIZE
    top  = cy * CHUNK_SIZE
    margin = 20

    # intersect chunk rect with world rect (leave a small margin from borders)
    minx = max(left + margin, WORLD_LEFT + margin)
    maxx = min(left + CHUNK_SIZE - margin, WORLD_RIGHT - margin)
    miny = max(top  + margin, WORLD_TOP  + margin)
    maxy = min(top  + CHUNK_SIZE - margin, WORLD_BOTTOM - margin)

    if minx >= maxx or miny >= maxy:
        return  # chunk lies outside world

    for _ in range(FOOD_PER_CHUNK):
        fx = random.uniform(minx, maxx)
        fy = random.uniform(miny, maxy)
        fr = FOOD_R
        kind = "boost" if random.random() < BOOST_FRACTION else "normal"
        foods.append({"x": fx, "y": fy, "r": fr, "kind": kind})

# ---------------- Spawn Balancer (per chunk) ----------------
TARGET_PER_CHUNK = FOOD_PER_CHUNK  # desired items per active chunk
MIN_BOOST_PER_CHUNK = 1            # guarantee at least one boost per active chunk

def spawn_items_in_chunk(cx: int, cy: int, n: int, kind: str) -> None:
    """Spawn exactly n items of given kind ('normal' or 'boost') inside chunk, respecting world bounds."""
    if n <= 0:
        return
    left = cx * CHUNK_SIZE
    top  = cy * CHUNK_SIZE
    margin = 20
    # intersect chunk rect with world rect
    minx = max(left + margin, WORLD_LEFT + margin)
    maxx = min(left + CHUNK_SIZE - margin, WORLD_RIGHT - margin)
    miny = max(top  + margin, WORLD_TOP  + margin)
    maxy = min(top  + CHUNK_SIZE - margin, WORLD_BOTTOM - margin)
    if minx >= maxx or miny >= maxy:
        return
    for _ in range(n):
        fx = random.uniform(minx, maxx)
        fy = random.uniform(miny, maxy)
        foods.append({"x": fx, "y": fy, "r": FOOD_R, "kind": kind})

SPAWN_RADIUS_CHUNKS = 1
SPAWN_INTERVAL = 0.9
last_density_spawn = 0.0

def periodic_spawn_around(points: list[tuple[float, float]]) -> None:
    """Every SPAWN_INTERVAL seconds, ensure each chunk near `points`
    reaches TARGET_PER_CHUNK total items and at least MIN_BOOST_PER_CHUNK boosts."""
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

    # Count totals and boosts per key
    total_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    boost_counts: dict[tuple[int, int], int] = {k: 0 for k in keys}
    for f in foods:
        k = chunk_of(f["x"], f["y"])
        if k in total_counts:
            total_counts[k] += 1
            if f.get("kind") == "boost":
                boost_counts[k] += 1

    # Spawn deficits: guarantee at least MIN_BOOST_PER_CHUNK boosts
    for key in keys:
        total = total_counts[key]
        boosts = boost_counts[key]
        desired_total = TARGET_PER_CHUNK
        # Aim for fraction, but guarantee a minimum of one boost
        desired_boosts = max(MIN_BOOST_PER_CHUNK, int(round(desired_total * BOOST_FRACTION)))
        missing_total = max(0, desired_total - total)
        missing_boosts = max(0, desired_boosts - boosts)

        if missing_boosts > 0:
            spawn_items_in_chunk(key[0], key[1], missing_boosts, "boost")
            missing_total -= missing_boosts

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
            snake.grow(FOOD_GROWTH)
            if f.get("kind") == "boost":
                snake.apply_boost(BOOST_MULT, BOOST_DURATION)
        else:
            keep.append(f)
    foods = keep
    return eaten

def draw_foods(surf: pygame.Surface, camx: float, camy: float) -> None:
    for f in foods:
        sx, sy = world_to_screen(f["x"], f["y"], camx, camy)
        col = BOOST_COLOR if f.get("kind") == "boost" else FOOD_COLOR
        pygame.draw.circle(surf, col, (sx, sy), f["r"])

def draw_world_border(surf: pygame.Surface, camx: float, camy: float) -> None:
    tx, ty = world_to_screen(WORLD_LEFT, WORLD_TOP, camx, camy)
    rect = pygame.Rect(tx, ty, WORLD_W, WORLD_H)
    pygame.draw.rect(surf, (80, 80, 80), rect, 2)

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
    # Predation rule: larger can eat smaller; smaller cannot eat larger
    if attacker.length <= PREDATION_RATIO * defender.length:
        return 0.0

    p0 = attacker.points[0]
    p1 = attacker.points[1]
    last_idx = len(defender.points) - 1

    # skip a few most‑recent defender segments near its head
    for i in range(skip_recent, last_idx):
        hit, ip = segment_intersection(p0, p1, defender.points[i], defender.points[i + 1])
        if hit:
            stolen = defender.cut_from_index(i, ip)
            attacker.length += stolen
            return stolen
    return 0.0

# ---------------- HUD ----------------
def draw_hud_one(surf: pygame.Surface, s: Snake, eaten: int, label: str = "") -> None:
    text = f"{label} LEN {int(s.length):4d}   SPD {int(s.speed_current):3d}   FOOD {eaten:3d}".strip()
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
    text1 = f"P1 LEN {int(s1.length):4d}   SPD {int(s1.speed_current):3d}   FOOD {eaten1:3d}"
    txt1 = FONT.render(text1, True, HUD_TEXT)
    pad = 10
    w1 = txt1.get_width() + pad * 2
    h1 = txt1.get_height() + pad * 2
    hud1 = pygame.Surface((w1, h1), pygame.SRCALPHA)
    pygame.draw.rect(hud1, HUD_BG, hud1.get_rect(), border_radius=10)
    hud1.blit(txt1, (pad, pad))
    surf.blit(hud1, (12, 10))

    # right panel (P2)
    text2 = f"P2 LEN {int(s2.length):4d}   SPD {int(s2.speed_current):3d}   FOOD {eaten2:3d}"
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
    global total_eaten1, total_eaten2
    running = True
    game_state = "menu"
    game_mode = None  # "1p" or "2p"

    # Two buttons centered with a small gap
    btn_w, btn_h, gap = 260, 64, 40
    left_x = (W - (btn_w * 2 + gap)) // 2
    btn1_rect = pygame.Rect(left_x, int(H * 0.55), btn_w, btn_h)
    btn2_rect = pygame.Rect(left_x + btn_w + gap, int(H * 0.55), btn_w, btn_h)

    while running:
        dt = clock.tick(60) / 1000.0

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

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
            draw_hud_one(screen, snake1, total_eaten1, "P1")
            pygame.display.flip()
            continue

        # --- 2P split-screen ---
        # Spawn around both; cull around midpoint
        ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
        ensure_chunks_around(snake2.x, snake2.y, radius_chunks=1)
        midx = 0.5 * (snake1.x + snake2.x)
        midy = 0.5 * (snake1.y + snake2.y)
        cull_far_foods(midx, midy, keep_radius_chunks=2)

        # Update both
        snake1.update(dt)
        snake2.update(dt)
        total_eaten1 += eat_food_if_colliding(snake1)
        total_eaten2 += eat_food_if_colliding(snake2)

        # Steal mechanic both ways (still respects predation)
        _ = steal_if_cross(snake1, snake2)
        _ = steal_if_cross(snake2, snake1)

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
        draw_hud_one(left, snake1, total_eaten1, "P1")

        # Right view (P2)
        right.fill(BG_COLOR)
        draw_world_border(right, cam2x, cam2y)
        draw_foods(right, cam2x, cam2y)
        snake1.draw(right, cam2x, cam2y)
        snake2.draw(right, cam2x, cam2y)
        draw_hud_one(right, snake2, total_eaten2, "P2")

        # Compose to screen
        screen.blit(left, (0, 0))
        screen.blit(right, (view_w, 0))
        pygame.draw.rect(screen, (30, 30, 30), pygame.Rect(view_w - 1, 0, 2, H))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()