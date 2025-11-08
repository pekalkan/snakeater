"""snakeater – Step 2: infinite world (camera) + food & growth"""

import math
import random
import pygame

# ---------------- Window / Pygame ----------------
pygame.init()
W, H = 960, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - step 2 (infinite world + food)")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 18)

# ---------------- Helpers ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def world_to_screen(px, py, camx, camy):
    return int(px - camx), int(py - camy)

# ---------------- Snake ----------------
class Snake:
    """Snake head + trail rendered along the recent path (length-limited)."""

    def __init__(self, x, y):
        # Kinematics
        self.x, self.y = x, y
        self.vx = 0.0
        self.vy = 0.0
        self.base_speed = 120.0
        # Heading + current speed (cruise vs boost)
        self.heading_x, self.heading_y = 1.0, 0.0   # default look direction (right)
        self.speed_current = self.base_speed

        # Body/trail
        self.thickness = 12              # radius of body circles
        self.length = 250.0              # total trail length in pixels
        self.min_step = 2.0              # add a new trail point if moved at least this much
        self.points = [(x, y)]           # trail points, head at index 0

    def handle_input(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        mag = math.hypot(dx, dy)

        if mag:  # input present → update heading, move 2x speed
            self.heading_x, self.heading_y = dx / mag, dy / mag
            self.speed_current = self.base_speed * 2.0
        else:    # no input → cruise at base speed
            self.speed_current = self.base_speed

        # velocity follows current heading
        self.vx = self.heading_x * self.speed_current
        self.vy = self.heading_y * self.speed_current

        # Infinite world (no clamping)
        self.x += self.vx * dt
        self.y += self.vy * dt

    def _trim_trail_to_length(self) -> None:
        """Ensure the accumulated trail length equals self.length by trimming the tail."""
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

    def grow(self, amount: float) -> None:
        self.length += float(amount)

    def update(self, dt: float) -> None:
        self.handle_input(dt)
        head = (self.x, self.y)
        if dist(head, self.points[0]) >= self.min_step:
            self.points.insert(0, head)
        self._trim_trail_to_length()

    def draw(self, surf: pygame.Surface, camx: float, camy: float) -> None:
        # Draw body: circles along the trail (skip every other point for performance)
        for p in self.points[::2]:
            sx, sy = world_to_screen(p[0], p[1], camx, camy)
            pygame.draw.circle(surf, (120, 200, 255), (sx, sy), self.thickness)
        # Head highlight
        hx, hy = world_to_screen(self.x, self.y, camx, camy)
        pygame.draw.circle(surf, (160, 230, 255), (hx, hy), self.thickness + 1)

# ---------------- Infinite World (chunks) ----------------
CHUNK_SIZE = 800                   # world is split into square chunks
FOOD_PER_CHUNK = 12
FOOD_R = 6
FOOD_GROWTH = 30.0                 # how much length a single food grants

spawned_chunks = set()             # {(cx, cy)}
foods = []                         # list of dicts: {"x","y","r"}

def chunk_of(px: float, py: float):
    # Python's // works with negatives as floor division; adjust to int
    cx = math.floor(px / CHUNK_SIZE)
    cy = math.floor(py / CHUNK_SIZE)
    return int(cx), int(cy)

def spawn_chunk(cx: int, cy: int) -> None:
    """Create FOOD_PER_CHUNK food items randomly within a chunk."""
    global foods
    left = cx * CHUNK_SIZE
    top = cy * CHUNK_SIZE
    margin = 20
    for _ in range(FOOD_PER_CHUNK):
        fx = random.uniform(left + margin, left + CHUNK_SIZE - margin)
        fy = random.uniform(top + margin, top + CHUNK_SIZE - margin)
        fr = FOOD_R
        foods.append({"x": fx, "y": fy, "r": fr})

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
    """Optionally remove foods too far away to keep memory usage bounded."""
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
        else:
            keep.append(f)
    foods = keep
    return eaten

def draw_foods(surf: pygame.Surface, camx: float, camy: float) -> None:
    for f in foods:
        sx, sy = world_to_screen(f["x"], f["y"], camx, camy)
        pygame.draw.circle(surf, (255, 205, 120), (sx, sy), f["r"])

# ---------------- HUD ----------------
def draw_hud(surf: pygame.Surface, snake: Snake, eaten_total: int) -> None:
    msg = f"Len: {int(snake.length)}   Speed: {int(snake.speed_current)}   Food eaten: {eaten_total}"
    surf.blit(FONT.render(msg, True, (240, 240, 240)), (12, 10))

# ---------------- Main Loop ----------------
snake = Snake(0.0, 0.0)          # start at world origin
total_eaten = 0

def main():
    global total_eaten
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        # World management
        ensure_chunks_around(snake.x, snake.y, radius_chunks=1)
        cull_far_foods(snake.x, snake.y, keep_radius_chunks=2)

        # Update
        snake.update(dt)
        total_eaten += eat_food_if_colliding(snake)

        # Camera follows the snake head
        camx = snake.x - W / 2
        camy = snake.y - H / 2

        # Draw
        screen.fill((18, 22, 28))
        draw_foods(screen, camx, camy)
        snake.draw(screen, camx, camy)
        draw_hud(screen, snake, total_eaten)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()