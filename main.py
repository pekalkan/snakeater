    def draw(self, surf: pygame.Surface, camx: float, camy: float) -> None:
        # Render a continuous tube: thick polyline + rounded caps
        path = [world_to_screen(p[0], p[1], camx, camy) for p in self.points]
        if len(path) >= 2:
            w = max(1, int(self.thickness * 2))
            # Thick body
            pygame.draw.lines(surf, SNAKE_BODY, False, path, w)
            # Anti-aliased outline to soften edges
            pygame.draw.aalines(surf, SNAKE_BODY, False, path)
            # Tail rounded cap
            pygame.draw.circle(surf, SNAKE_BODY, path[-1], self.thickness)

        # Head rounded cap + highlight
        hx, hy = world_to_screen(self.x, self.y, camx, camy)
        pygame.draw.circle(surf, SNAKE_HEAD, (hx, hy), self.thickness + 2)