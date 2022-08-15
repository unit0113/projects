import pyaudio
import numpy as np
from math import sqrt
import pygame

pygame.init()

WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
FPS = 60
BLACK = (0, 0, 0)

RATE = 44100
CHUNK = int(RATE / FPS)
FORMAT = pyaudio.paInt16
LINE_WIDTH = int(WIDTH/CHUNK)


def initialize():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Audio Visualizer")

    py_audio = pyaudio.PyAudio()

    stream = py_audio.open(format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        output_device_index = 3,
        frames_per_buffer=CHUNK)

    return screen, stream


def main():
    screen, stream = initialize()
    clock = pygame.time.Clock()

    pygame.mixer.init()
    pygame.mixer.music.load('Python_Playground\song_visualizer\Christopher Tin - The Storm-Driven Sea.mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            pygame.quit()
            quit()

        if keys[pygame.K_r]:
            main()

        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        fft_complex = np.fft.fft(data, n=CHUNK)

        screen.fill(BLACK)
        color = (0,128,1)

        #max_val = sqrt(max(v.real * v.real + v.imag * v.imag for v in fft_complex))
        #scale_value = HEIGHT / max_val
        for i, v in enumerate(fft_complex):
            #v = complex(v.real / dist1, v.imag / dist1)
            dist = sqrt(v.real * v.real + v.imag * v.imag)
            #mapped_dist = dist * scale_value
        
            pygame.draw.line(screen, color, (i * LINE_WIDTH, HEIGHT), (i * LINE_WIDTH, HEIGHT - dist), LINE_WIDTH)

        pygame.display.update()


if __name__ == "__main__":
    main()
