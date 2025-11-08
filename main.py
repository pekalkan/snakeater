"""snakeater – continuous tube snake, infinite world, food + boost"""

import math
import random
import time
import pygame

# ---------------- Window / Pygame ----------------
pygame.init()
W, H = 1920, 1080
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - step 2 (infinite world + food)")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Menlo", 20)

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

# ---------------- Helpers ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def world_to_screen(px, py, camx, camy):
    return int(px - camx), int(py - camy)

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

BOOST_FRACTION = 0.02   # 2% of foods are boost orbs
BOOST_MULT     = 1.5    # move at 1.5x when boosted
BOOST_DURATION = 5.0    # boost lasts 5 seconds

spawned_chunks = set()  # {(cx, cy)}
foods = []              # list of dicts: {"x","y","r","kind"}

def chunk_of(px: float, py: float):
    cx = math.floor(px / CHUNK_SIZE)
    cy = math.floor(py / CHUNK_SIZE)
    return int(cx), int(cy)

def spawn_chunk(cx: int, cy: int) -> None:
    """Create FOOD_PER_CHUNK food items randomly within a chunk.
    Each food has BOOST_FRACTION chance to be a boost orb."""
    global foods
    left = cx * CHUNK_SIZE
    top  = cy * CHUNK_SIZE
    margin = 20

    for _ in range(FOOD_PER_CHUNK):
        fx = random.uniform(left + margin, left + CHUNK_SIZE - margin)
        fy = random.uniform(top + margin, top + CHUNK_SIZE - margin)
        fr = FOOD_R
        kind = "boost" if random.random() < BOOST_FRACTION else "normal"
        foods.append({"x": fx, "y": fy, "r": fr, "kind": kind})

def ensure_chunks_around(px: float, py: float, radius_chunks: int = 1) -> None:
    """Make sure the 3x3 neighborhood around the player is spawned."""
    pcx, pcy = chunk_of(px, py)
    for dy in range(-radius_chunks, radius_chunks + 1):
        for dx in range(-radius_chunks, radius_chunks + 1):
            key = (pcx + dx, pcy + dy)
            if key not in spawned_chunks:
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


# ---------------- Steal Mechanic ----------------
def steal_if_cross(attacker: Snake, defender: Snake, skip_recent: int = 4) -> float:
    """If the attacker's newest head segment crosses any older segment
    of the defender, cut the defender there and transfer the removed
    length to the attacker. Returns the stolen length (0.0 if none)."""
    if len(attacker.points) < 2 or len(defender.points) < 2:
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
    while running:
        dt = clock.tick(60) / 1000.0

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        # World management
        ensure_chunks_around(snake1.x, snake1.y, radius_chunks=1)
        ensure_chunks_around(snake2.x, snake2.y, radius_chunks=1)
        # cull around midpoint so both players' vicinity stays populated
        midx = 0.5 * (snake1.x + snake2.x)
        midy = 0.5 * (snake1.y + snake2.y)
        cull_far_foods(midx, midy, keep_radius_chunks=2)

        # Update
        snake1.update(dt)
        snake2.update(dt)
        total_eaten1 += eat_food_if_colliding(snake1)
        total_eaten2 += eat_food_if_colliding(snake2)

        # Steal mechanic: if one crosses the other, transfer length
        stolen12 = steal_if_cross(snake1, snake2)
        stolen21 = steal_if_cross(snake2, snake1)

        # Camera follows midpoint between snakes
        camx = 0.5 * (snake1.x + snake2.x) - W / 2
        camy = 0.5 * (snake1.y + snake2.y) - H / 2

        # Draw
        screen.fill(BG_COLOR)
        draw_foods(screen, camx, camy)
        snake1.draw(screen, camx, camy)
        snake2.draw(screen, camx, camy)
        draw_hud_two(screen, snake1, total_eaten1, snake2, total_eaten2)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()