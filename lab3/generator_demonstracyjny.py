"""Generator zastępczy do demonstracji, a nie generator prowadzącego.

Stałe ustalono przed oceną testową. Algorytm detect_pair() nie importuje
ani nie odczytuje parametrów tego modułu. Po otrzymaniu generatora z zajęć
należy zastąpić wywołanie w sekcji A notebooka i powtórzyć eksperyment.
"""

import numpy as np


def generate_interference(images, seed):
    """Dodaj wspólną sinusoidę wzdłuż kolumn, z osobną fazą dla obrazu.

    Wejście i wyjście: RGB NHWC, skala 0-1, bez obcinania wyjścia.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Oczekiwano obrazów RGB w układzie NHWC.')
    amplitude = 0.25
    frequency = 7 / 32
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2 * np.pi, size=(len(images), 1, 1, 1))
    columns = np.arange(images.shape[2])[None, None, :, None]
    noise = amplitude * np.sin(2 * np.pi * frequency * columns + phase)
    return (images + noise).astype(np.float32)
