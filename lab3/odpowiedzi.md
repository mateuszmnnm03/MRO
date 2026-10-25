# Laboratorium 3

## 1. Jakie współrzędne znaleziono i dlaczego tworzą parę?

**(16, 10) i (16, 22)** - Leżą symetrycznie względem środka (16, 16), po 6 kolumn w obie strony. Tworzą parę, ponieważ rzeczywista sinusoida jest reprezentowana w FFT przez dwie składowe o przeciwnych częstotliwościach.

## 2. Jakie są wyniki i zysk accuracy?

| Wariant   | Accuracy | F1-macro |     MSE |
| --------- | -------: | -------: | ------: |
| Czyste    |   41,80% |    0,405 | 0,00000 |
| Zakłócone |   34,30% |    0,304 | 0,02000 |
| Notch     |   41,80% |    0,406 | 0,00013 |
| Gauss     |   37,10% |    0,371 | 0,01194 |

Względem zakłóconych obrazów notch daje **+7,50 p.p.**, a Gauss **+2,80 p.p.** Notch osiąga taką samą accuracy jak dla czystych danych.

## 3. Dlaczego nie można odtworzyć obrazu z samej amplitudy? Co stałoby się przy uśrednieniu zespolonych widm?

Amplituda nie zawiera fazy, która określa przesunięcie. Bez niej nie odtworzymy wiernie jego struktury. Jeśli uśrednialibyśmy bezpośrednio zespolone wartości FFT, zakłócenia o różnych losowych fazach mogłyby się wzajemnie znosić.

## 4. Dlaczego notch pomaga bardziej i czy może zaszkodzić?

Zakłócenie skupia się w dwóch częstotliwościach, więc notch odpowiednio je usuwa. Gauss rozmywa także potrzebne krawędzie i tekstury, potrzebne do rozpoznawania. Notch również może zaszkodzić, usuwając prawdziwe detale o tych samych częstotliwościach.
