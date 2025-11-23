# renderer_farol.py
import pygame
from environments.environment import Enviroment

VAZIO = 0
PAREDE = 1
FAROL = 2

class FarolRenderer:
    def __init__(self, env: Enviroment, cell_size=25):
        pygame.init()

        self.env = env
        self.cell_size = cell_size

        w, h = env.tamanho
        self.width = w * cell_size
        self.height = h * cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Farol Environment")

        self.clock = pygame.time.Clock()
        self.running = True

        # cores
        self.colors = {
            VAZIO: (230, 230, 230),   # cinza claro
            PAREDE: (50, 50, 50),     # cinza escuro
            FAROL: (255, 220, 0),     # amarelo farol
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

        for y in range(h):
            for x in range(w):
                cell = self.env.mapa_estado[y, x]
                color = self.colors.get(cell, (255, 0, 0))

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                )

        # desenhar agentes
        for pos in agent_positions.values():
            ax, ay = pos
            pygame.draw.rect(
                self.screen,
                self.colors["agente"],
                pygame.Rect(
                    ax * self.cell_size,
                    ay * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
            )

        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS = 1 movimento a cada 100ms

        return True

    def close(self):
        pygame.quit()