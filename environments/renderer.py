import pygame
from environments.environment import Enviroment

VAZIO = 0
PAREDE = 1
OBJETIVO = 2

class EnvRenderer:
    def __init__(self, env: Enviroment, window_size=600):
        pygame.init()

        self.env = env

        w, h = env.tamanho

        # janela fixa (900x900 por default)
        self.window_size = window_size
        self.cell_size = max(1, window_size // max(w, h))

        # cria janela fixa
        self.width = w * self.cell_size
        self.height = h * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Environment")

        self.clock = pygame.time.Clock()
        self.running = True

        # cores
        self.colors = {
            VAZIO: (230, 230, 230),   # cinza claro
            PAREDE: (50, 50, 50),     # cinza escuro
            OBJETIVO: (255, 220, 0),     # amarelo farol
            "agente": (0, 150, 255)   # azul agente
        }

    def draw(self, agent_positions):
        """
        Desenha o mapa e os agentes.
        agent_positions deve ser: {agent_id: (x,y)}
        """

        # gerir eventos (fechar janela)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

        # desenhar mapa
        self.screen.fill((0, 0, 0))

        h, w = self.env.mapa_estado.shape
        cs = self.cell_size

        for y in range(h):
            for x in range(w):
                cell = self.env.mapa_estado[y, x]
                color = self.colors.get(cell, (255, 0, 0))

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        x * cs,
                        y * cs,
                        cs,
                        cs
                    )
                )

        # desenhar agentes
        for pos in agent_positions.values():
            ax, ay = pos
            pygame.draw.rect(
                self.screen,
                self.colors["agente"],
                pygame.Rect(
                    ax * cs,
                    ay * cs,
                    cs,
                    cs
                )
            )

        pygame.display.flip()
        self.clock.tick(12)  # 10 FPS

        return True

    def close(self):
        pygame.quit()