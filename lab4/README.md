# Laboratorium 4 - tekstury i falki

Główny plik: [Laboratorium_4.ipynb](Laboratorium_4.ipynb). Realizuje punkty A-E przekazane w treści zadania. Notebook został wykonany; zawiera wyniki i wykresy.

- Pełny Kylberg v1.0: 28 klas po 160 obrazów, oryginalne PNG 576 x 576, bez gotowych obrotów.
- Wybrane klasy: blanket1, canvas1, cushion1, lentils1, rice1, stone1.
- Mały S1: 40 obrazów na klasę (20 trening / 10 walidacja / 10 test), razem 240.
- Duży S1: 160 obrazów na klasę (80 trening / 40 walidacja / 40 test), razem 960.
- Podział według fotografii źródłowych: a,b do uczenia, c do walidacji, d do testu. Mały podzbiór jest zawarty w dużym.
- Dwie CNN z dropout i augmentacją wyłącznie treningową; wybór epoki według straty walidacyjnej.
- S2 dla Haar i db2, po 3 poziomy, zapis pełnych współczynników w oryginalnej rozdzielczości.
- S3: dokładnie 9 średnich energii, w kolejności H1,V1,D1,H2,V2,D2,H3,V3,D3. Bez aproksymacji.
- Oba modele porównywane na tym samym pełnym teście 240 obrazów (grupa d); dodatkowo wynik na wspólnym małym teście 60 obrazów. Liczebności walidacji są różne, co jest zaznaczone przy interpretacji.

## Uruchomienie

Notebook można otworzyć z katalogu `lab4` albo z katalogu głównego projektu. Wybierz kernel z pakietami z `requirements-lab4.txt`; w obecnym projekcie wykorzystano środowisko `lab2/.venv`.

Pobranie danych w razie potrzeby:

```powershell
& ../lab2/.venv/Scripts/python.exe pobierz_dane.py
```

Następnie wykonaj notebook od początku. Istniejące zgodne pliki S2/S3 i modele są wczytywane; przełącznik `RETRAIN` pozwala wymusić nowy trening. Dane i dwa pełne zbiory współczynników zajmują kilka GB. Duże pliki są pomijane przez lokalny `.gitignore`; manifest pobrań, podział, metryki i wektory 9D pozostają zapisane.

Pojedynczy przebieg jest wstępnym eksperymentem, bez twierdzenia o istotności statystycznej lub najlepszej falce. Dodatkowy klasyfikator liniowy na 9D jest kontrolą użyteczności tych cech, wykraczającą poza obowiązkowe utworzenie S3.

## Pierwsze wyniki

Accuracy na tym samym teście 240 obrazów, po 40 na każdą z 6 klas:

| Metoda | Mały trening: 120 obrazów | Duży trening: 480 obrazów |
|---|---:|---:|
| CNN | 97,08% | 100,00% |
| 9D Haar + klasyfikator liniowy | 93,33% | 95,00% |
| 9D db2 + klasyfikator liniowy | 90,00% | 89,58% |

100% dotyczy tego podziału i 6 wybranych klas. Dla małej CNN wszystkie 7 błędów to cushion1 rozpoznane jako lentils1. Większy zbiór nie poprawił każdej metody: db2 straciła jeden poprawnie rozpoznany obraz. Dokładne wyniki, F1-macro i ograniczenia: `wyniki_lab4/podsumowanie.json`; predykcje: `wyniki_lab4/predykcje.npz`.

## Źródła

- G. Kylberg, *The Kylberg Texture Dataset v. 1.0*, External report No. 35, 2011: https://user.it.uu.se/~gusky180/texture/
- Serwer autora z oryginalnymi archiwami: https://filedn.com/lkCRue0RhPO7ercIrqFRl2Y/datasets/KylbergTextureDatasetV1/without-rotations-zip/
- Dokumentacja `wavedec2` i kolejności współczynników: https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html

Treść zadania odwołuje się do niedostarczonego rysunku/wzoru energii. Przyjęto energię średnią E = mean(c**2), zgodnie z opisem słownym; nie stosuje się mean(c) ani mean(abs(c)).
