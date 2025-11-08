# snakeater - Step 1: pencere + yılan gibi gövde
import math, pygame
pygame.init()
W, H = 960, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("snakeater - step 1")
clock = pygame.time.Clock()

def clamp(v, lo, hi): return max(lo, min(hi, v))
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

class Snake:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.speed = 240.0
        self.thickness = 12
        self.length = 250.0
        self.min_step = 2.0
        self.points = [(x, y)]

    def input(self, dt):
        k = pygame.key.get_pressed()
        dx = (k[pygame.K_d] or k[pygame.K_RIGHT]) - (k[pygame.K_a] or k[pygame.K_LEFT])
        dy = (k[pygame.K_s] or k[pygame.K_DOWN]) - (k[pygame.K_w] or k[pygame.K_UP])
        mag = math.hypot(dx, dy)
        if mag: dx, dy = dx/mag, dy/mag
        self.vx, self.vy = dx*self.speed, dy*self.speed
        self.x = clamp(self.x + self.vx*dt, self.thickness, W-self.thickness)
        self.y = clamp(self.y + self.vy*dt, self.thickness, H-self.thickness)

    def trim(self):
        total = 0.0
        for i in range(len(self.points)-1):
            total += dist(self.points[i], self.points[i+1])
        while total > self.length and len(self.points) > 1:
            a, b = self.points[-2], self.points[-1]
            seg = dist(a, b)
            need = total - self.length
            if need >= seg:
                self.points.pop()
                total -= seg
            else:
                t = (seg - need)/seg if seg else 0
                self.points[-1] = (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)
                break

    def update(self, dt):
        self.input(dt)
        head = (self.x, self.y)
        if dist(head, self.points[0]) >= self.min_step:
            self.points.insert(0, head)
        self.trim()

    def draw(self, s):
        for p in self.points[::2]:
            pygame.draw.circle(s, (120,200,255), (int(p[0]), int(p[1])), self.thickness)
        pygame.draw.circle(s, (160,230,255), (int(self.x), int(self.y)), self.thickness+1)

snake = Snake(W/2, H/2)

def main():
    run = True
    while run:
        dt = clock.tick(60)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                run = False
        snake.update(dt)
        screen.fill((18,22,28))
        snake.draw(screen)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()