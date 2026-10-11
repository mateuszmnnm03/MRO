"""Buduje notebook z opisem i kodem; nie uruchamia eksperymentu."""
from pathlib import Path
import json
import sys
import textwrap

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


md(r"""
# Laboratorium 2 — reprezentacje obrazów w sieci konwolucyjnej

**Zbiór:** MNIST, czyli obrazy cyfr 0–9 w skali szarości, o rozmiarze 28 × 28 pikseli.
**Narzędzia:** PyTorch, NumPy, Matplotlib i scikit-learn.

Notebook realizuje wymagania z `zadanie.txt` i zawiera dodatkowy eksperyment porównawczy:

1. Przygotowanie danych oraz pokazanie tych samych obrazów w wariantach A, B i C.
2. Sieć z częścią konwolucyjną, `Flatten`, **dokładnie trzema ukrytymi warstwami gęstymi** i osobnym wyjściem klasyfikacyjnym.
3. Osobny trening takiej samej architektury dla każdego wariantu.
4. Porównanie jakości klasyfikacji oraz reprezentacji obrazów metodą t-SNE.
5. Wyjaśnienia operacji, interpretacja wyników i ograniczenia doświadczenia.

**Zakres:** opis informuje, że dalsze cele będą przedstawione na zajęciach. Ten notebook obejmuje wszystkie podane wymagania; dodatkowe polecenia prowadzącego można dopisać na końcu. Wyniki liczbowe w komórkach wyjściowych pochodzą z wykonania kodu, a przewidywania teoretyczne są oznaczone jako przewidywania.
""")

md(r"""
## 1. Jak uruchomić notebook

Otwórz plik w Jupyter, VS Code albo Google Colab i wykonuj komórki od góry (`Run All`). W tym projekcie przygotowano lokalne środowisko **`.venv` z Pythonem 3.12**: w VS Code wybierz jako kernel `.venv/Scripts/python.exe`. Domyślny Python 3.14 korzysta z innego zestawu bibliotek. Przy przenoszeniu notebooka na inny komputer dołączony plik `requirements-lab2.txt` zawiera zależności.

W razie brakujących pakietów odkomentuj `%pip` poniżej, wykonaj komórkę i zrestartuj kernel. Instalacja używa środowiska aktualnie wybranego kernela. Pierwsze pobranie MNIST wymaga internetu; kolejne uruchomienia korzystają z katalogu `data/`.

Domyślnie uczymy na 20 000 przykładów przez 5 epok, walidujemy na 5 000 i testujemy na wszystkich 10 000 obrazach testowych. To krótki, powtarzalny eksperyment. Zmień konfigurację, jeśli masz więcej czasu. GPU jest używane automatycznie, jeśli PyTorch je wykryje.
""")
code(r"""
# Tylko jeśli brakuje bibliotek — następnie zrestartuj kernel:
# %pip install torch torchvision numpy matplotlib "scikit-learn>=1.5,<2"
""")
code(r"""
import copy
import csv
import inspect
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from IPython.display import Markdown, display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST

SEED = 42
N_TRAIN = 20_000          # None: wszystkie obrazy pozostałe po wydzieleniu walidacji
N_VAL = 5_000
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TSNE_SAMPLES = 1_000     # te same indeksy obrazów testowych dla A, B i C
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 750
DATA_DIR = Path("data")
OUTPUT_DIR = Path("wyniki_lab2")
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, sklearn {sklearn.__version__}")
print("Urządzenie:", DEVICE)
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
print("Ziarno:", SEED)
""")
md(r"""
## 2. Dane, etykiety i podział

Każdy obraz zawiera 784 liczby całkowite z przedziału 0–255. Wartość 0 oznacza czerń, a 255 biel. Etykieta jest liczbą od 0 do 9 i mówi, jaka cyfra znajduje się na obrazie.

MNIST zawiera 60 000 obrazów treningowych i 10 000 testowych. Ze zbioru treningowego wydzielamy walidację, a następnie wybieramy podzbiór do krótkiego treningu. Podziały są **stratyfikowane**, czyli zachowują w przybliżeniu proporcje klas. Zbiór testowy służy dopiero do końcowej oceny. Dobór najlepszej epoki opiera się wyłącznie na walidacji.

Obrazy przechowujemy jako `uint8`, aby oszczędzić pamięć. Dopiero przy pobieraniu próbki do sieci dzielimy wartości przez 255. Nie obliczamy statystyk ze zbioru testowego.
""")
code(r"""
train_mnist = MNIST(root=str(DATA_DIR), train=True, download=True)
test_mnist = MNIST(root=str(DATA_DIR), train=False, download=True)
x_train_original = train_mnist.data.unsqueeze(1)  # [N, C=1, H=28, W=28]
x_test_original = test_mnist.data.unsqueeze(1)
y_train = train_mnist.targets.clone()
y_test = test_mnist.targets.clone()

all_ids = np.arange(len(y_train))
train_ids, val_ids = train_test_split(
    all_ids, test_size=N_VAL, stratify=y_train.numpy(), random_state=SEED
)
if N_TRAIN is not None and N_TRAIN < len(train_ids):
    train_ids, _ = train_test_split(
        train_ids, train_size=N_TRAIN,
        stratify=y_train.numpy()[train_ids], random_state=SEED
    )
elif N_TRAIN is not None and N_TRAIN > len(train_ids):
    raise ValueError("N_TRAIN przekracza liczbę dostępnych obrazów treningowych.")

assert set(train_ids).isdisjoint(set(val_ids))
print("Pełne dane:", tuple(x_train_original.shape), tuple(x_test_original.shape))
print(f"Trening: {len(train_ids)}, walidacja: {len(val_ids)}, test: {len(y_test)}")
print("Klasy w treningu:", np.bincount(y_train.numpy()[train_ids], minlength=10))
""")
md(r"""
## 3. Warianty A, B i C — co dokładnie przestawiamy?

| Wariant | Sposób przygotowania | Co pozostaje? |
|---|---|---|
| A | Obraz oryginalny | Lokalny kształt cyfry oraz położenie każdego piksela. |
| B | Jedna permutacja, wspólna dla treningu, walidacji i testu | Stałe powiązanie nowych pozycji ze starymi; wszystkie wartości pikseli. |
| C | Osobna permutacja dla każdego obrazu | Wszystkie wartości pikseli danego obrazu, ale brak wspólnego układu przestrzennego. |

**Permutacja** to przestawienie kolejności bez dodawania i usuwania elementów. Dla obrazu spłaszczonego do wektora: `nowy[j] = stary[permutacja[j]]`. Po przestawieniu przywracamy poprzedni rozmiar obrazu. Histogram jasności, suma i średnia pikseli nie zmieniają się.

W B losujemy kolejność raz. W C losujemy ją osobno dla każdego obrazu, **przed treningiem**, i zapisujemy gotowe obrazy w pamięci. `DataLoader` zmienia tylko kolejność przykładów, a nie przetasowanie ich pikseli. Osobne ziarna dla części treningowej i testowej C zapobiegają przypadkowemu powieleniu sekwencji permutacji.

Funkcja poniżej działa również dla obrazów kolorowych w formacie `[N, C, H, W]`: pozycje przestrzenne są przestawiane, a kanały jednego piksela wędrują razem. W eksperymencie korzystamy z jednego kanału MNIST.
""")
code(r"""
def permute_pixels(images, *, seed=None, permutation=None):
    "Wspólna permutacja, jeśli podano permutation; w innym przypadku osobna dla obrazu."
    if images.ndim != 4 or images.device.type != "cpu":
        raise ValueError("Oczekiwano tensora CPU [N, C, H, W].")
    n, channels, height, width = images.shape
    # Każdy wiersz spatial przechowuje wszystkie kanały jednego piksela.
    spatial = images.permute(0, 2, 3, 1).reshape(n, height * width, channels)
    if permutation is not None:
        permutation = np.asarray(permutation)
        if not np.array_equal(np.sort(permutation), np.arange(height * width)):
            raise ValueError("Niepoprawna permutacja pozycji przestrzennych.")
        moved = spatial[:, torch.as_tensor(permutation, dtype=torch.long), :]
    else:
        rng = np.random.default_rng(seed)
        moved = torch.empty_like(spatial)
        for i in range(n):
            order = torch.from_numpy(rng.permutation(height * width))
            moved[i] = spatial[i, order, :]
    return moved.reshape(n, height, width, channels).permute(0, 3, 1, 2).contiguous()

common_permutation = np.random.default_rng(SEED + 1).permutation(28 * 28)
variants = {
    "A": {"train": x_train_original, "test": x_test_original},
    "B": {
        "train": permute_pixels(x_train_original, permutation=common_permutation),
        "test": permute_pixels(x_test_original, permutation=common_permutation),
    },
    "C": {
        "train": permute_pixels(x_train_original, seed=SEED + 2),
        "test": permute_pixels(x_test_original, seed=SEED + 3),
    },
}
np.save(OUTPUT_DIR / "permutacja_B.npy", common_permutation)
np.savez(OUTPUT_DIR / "podzialy.npz", train_ids=train_ids, val_ids=val_ids)

# Kontrola zachowania rozmiaru i wszystkich wartości dla wybranych obrazów.
for name, data in variants.items():
    for split, original in (("train", x_train_original), ("test", x_test_original)):
        assert data[split].shape == original.shape
        assert torch.equal(
            data[split][:100].flatten(1).sort(dim=1).values,
            original[:100].flatten(1).sort(dim=1).values,
        )
inverse_B = torch.from_numpy(np.argsort(common_permutation))
assert torch.equal(
    variants["B"]["test"][:10].flatten(1)[:, inverse_B],
    x_test_original[:10].flatten(1),
)
assert torch.equal(
    variants["C"]["test"][:5], permute_pixels(x_test_original[:5], seed=SEED + 3)
)
# Miniaturowy przykład RGB: sprawdzamy, że trójki kanałów pozostały razem.
rgb_example = torch.arange(12).reshape(1, 3, 2, 2)
rgb_moved = permute_pixels(rgb_example, permutation=np.array([3, 1, 0, 2]))
assert torch.equal(rgb_moved[:, :, 0, 0], rgb_example[:, :, 1, 1])
print("Kontrole permutacji zakończone poprawnie. Etykiety są wspólne dla A, B i C.")
""")
code(r"""
# Jeden przykład każdej cyfry; indeksy są identyczne we wszystkich wierszach.
example_ids = [int(torch.where(y_test == digit)[0][0]) for digit in range(10)]
fig, axes = plt.subplots(3, 10, figsize=(14, 4.8))
for row, name in enumerate(("A", "B", "C")):
    for col, index in enumerate(example_ids):
        axes[row, col].imshow(variants[name]["test"][index, 0], cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f"Cyfra {int(y_test[index])}")
        if col == 0:
            axes[row, col].set_ylabel(name, rotation=0, labelpad=15, fontsize=14)
fig.suptitle("Te same obrazy testowe: oryginał, wspólna permutacja, osobne permutacje")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "obrazy_ABC.png", bbox_inches="tight")
plt.show()
""")
md(r"""
## 4. Architektura CNN + MLP

Konwolucja przesuwa mały filtr po obrazie i tworzy mapę cech. W A filtry mogą wykrywać np. krótkie odcinki i krawędzie. ReLU (`max(0, x)`) wprowadza nieliniowość. Max pooling wybiera największą wartość z okna 2 × 2, zmniejszając wymiary map. `Flatten` zmienia mapy w wektor, **bez uczenia i bez zmiany wartości**.

| Etap | Rozmiar pojedynczego przykładu | Znaczenie |
|---|---|---|
| Wejście | 1 × 28 × 28 | Obraz znormalizowany do [0, 1]. |
| Conv 3 × 3, 16 filtrów + ReLU + MaxPool | 16 × 14 × 14 | Pierwsze mapy cech; padding=1 zachowuje rozmiar przed poolingiem. |
| Conv 3 × 3, 32 filtry + ReLU + MaxPool | 32 × 7 × 7 | Kolejne cechy. |
| Flatten | 1568 | 32 · 7 · 7 liczb. |
| Ukryta gęsta 1 + ReLU | 128 | Pierwsza reprezentacja MLP. |
| Ukryta gęsta 2 + ReLU | 64 | Druga reprezentacja MLP. |
| Ukryta gęsta 3 + ReLU | 32 | Trzecia reprezentacja MLP. |
| **Osobna warstwa wyjściowa** | 10 | Logity klas 0–9. |

Warstwa gęsta oblicza `Wx + b` i może łączyć wszystkie współrzędne wejścia. W PyTorch odpowiada jej `nn.Linear`. Model ma trzy ukryte `Linear` i czwarty `Linear` jako wyjście. Funkcja `representations()` udostępnia aktywacje po ReLU, aby można było obejrzeć, jak zmienia się opis obrazów w kolejnych warstwach.

**Dlaczego bez softmax w `forward`?** `CrossEntropyLoss` przyjmuje logity i wykonuje odpowiednie przekształcenie wewnętrznie. Dla prawdopodobieństw do prezentacji można użyć `torch.softmax(logits, dim=1)`, a do wyboru klasy wystarczy `argmax`.
""")
code(r"""
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 10)

    def representations(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        conv = self.pool(self.relu(self.conv2(x)))
        flat = self.flatten(conv)
        h1 = self.relu(self.fc1(flat))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return {"Flatten": flat, "MLP 1": h1, "MLP 2": h2, "MLP 3": h3}

    def forward(self, x):
        return self.output(self.representations(x)["MLP 3"])

set_seed()
initial_model = DigitCNN()
initial_weights = copy.deepcopy(initial_model.state_dict())
assert sum(isinstance(layer, nn.Linear) for layer in initial_model.modules()) == 4
assert initial_model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
print(initial_model)
print("Liczba uczonych parametrów:", sum(p.numel() for p in initial_model.parameters()))
for name, activation in initial_model.representations(torch.zeros(2, 1, 28, 28)).items():
    print(name, tuple(activation.shape))
""")
md(r"""
## 5. Uczciwe porównanie i przebieg treningu

Trenujemy **trzy osobne modele**: model A na A, model B na B, a model C na C. Każdy jest oceniany na odpowiadającym mu wariancie testowym. To mierzy możliwość nauczenia się klasyfikacji w danej reprezentacji.

Wszystkie modele mają tę samą architekturę, identyczne początkowe wagi, podział danych, kolejność próbek w epokach oraz hiperparametry. Dla każdego tworzony jest nowy optymalizator Adam. Dzięki temu różnice nie wynikają ze startowania B lub C z wagami wyuczonymi na A. Ziarno poprawia odtwarzalność, lecz różne wersje bibliotek i urządzenia mogą dać nieco inne wyniki.

**Epoka** to jedno przejście po zbiorze treningowym. **Batch** to porcja przykładów przetwarzanych razem. Dla każdej porcji zerujemy gradienty, liczymy predykcję i stratę, wykonujemy `backward()` i aktualizujemy wagi. Accuracy to udział poprawnych predykcji, a cross-entropy ocenia również to, jak duży wynik model przypisuje prawdziwej klasie. Mniejsza strata jest lepsza.

W walidacji/testowaniu korzystamy z `model.eval()` i wyłączamy gradienty. Po każdej epoce zapisujemy statystyki. Wagi z najniższą stratą walidacyjną zostają przywrócone przed końcowym testem.
""")
code(r"""
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Tutaj nie losujemy permutacji — obrazy są już przygotowane.
        return self.images[index].to(torch.float32).div(255.0), self.labels[index]

def make_loaders(name):
    train_data = ImageDataset(variants[name]["train"], y_train)
    test_data = ImageDataset(variants[name]["test"], y_test)
    generator = torch.Generator().manual_seed(SEED)
    settings = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=DEVICE.type == "cuda")
    return (
        DataLoader(Subset(train_data, train_ids.tolist()), shuffle=True, generator=generator, **settings),
        DataLoader(Subset(train_data, val_ids.tolist()), shuffle=False, **settings),
        DataLoader(test_data, shuffle=False, **settings),
    )

criterion = nn.CrossEntropyLoss()

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count

@torch.inference_mode()
def predict(model, loader):
    model.eval()
    return np.concatenate([model(images.to(DEVICE)).argmax(1).cpu().numpy() for images, _ in loader])

models, histories, results, predictions = {}, {}, {}, {}
for name in ("A", "B", "C"):
    set_seed()
    train_loader, val_loader, test_loader = make_loaders(name)
    model = DigitCNN().to(DEVICE)
    model.load_state_dict(initial_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{name} | epoka {epoch}/{EPOCHS} | strata {train_loss:.4f} | "
              f"acc trening {train_accuracy:.2%} | acc walidacja {val_accuracy:.2%}", flush=True)
    model.load_state_dict(best_state)
    test_loss, test_accuracy = run_epoch(model, test_loader)
    predictions[name] = predict(model, test_loader)
    results[name] = {"test_loss": test_loss, "test_accuracy": test_accuracy,
                     "best_epoch": best_epoch, "seconds": time.perf_counter() - started}
    models[name], histories[name] = model, history
    torch.save(best_state, OUTPUT_DIR / f"model_{name}.pt")
    print(f"{name}: test {test_accuracy:.2%}, najlepsza epoka {best_epoch}\n", flush=True)
""")
md(r"""
## 6. Wyniki klasyfikacji

Krzywe treningu pokazują, czy model nadal się uczy. Jeżeli accuracy treningowa rośnie, a walidacyjna nie, może to wskazywać na przeuczenie. Strata treningowa jest średnią z kolejnych kroków aktualizacji w danej epoce; walidacyjna jest liczona dla wag na końcu epoki. Nie są to idealnie równoczesne pomiary.

Macierz pomyłek pokazuje wierszami prawdziwe cyfry, a kolumnami przewidywane. Przekątna oznacza poprawne decyzje. Wyniki poza przekątną mówią, które cyfry model myli. Pokazujemy odsetki wewnątrz każdej prawdziwej klasy (`normalize='true'`).
""")
code(r"""
table = "| Wariant | Accuracy test | Strata test | Najlepsza epoka | Czas [s] |\n"
table += "|---|---:|---:|---:|---:|\n"
for name, result in results.items():
    table += (f"| {name} | {result['test_accuracy']:.2%} | {result['test_loss']:.4f} | "
              f"{result['best_epoch']} | {result['seconds']:.1f} |\n")
display(Markdown(table))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, history in histories.items():
    epochs = [row["epoch"] for row in history]
    line, = axes[0].plot(epochs, [row["train_accuracy"] for row in history], label=f"{name}: trening")
    axes[0].plot(epochs, [row["val_accuracy"] for row in history], "--", color=line.get_color(), label=f"{name}: walidacja")
    line, = axes[1].plot(epochs, [row["train_loss"] for row in history], label=f"{name}: trening")
    axes[1].plot(epochs, [row["val_loss"] for row in history], "--", color=line.get_color(), label=f"{name}: walidacja")
for ax, ylabel in zip(axes, ("Accuracy", "Cross-entropy")):
    ax.set_xlabel("Epoka")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, EPOCHS + 1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "trening.png", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, name in zip(axes, ("A", "B", "C")):
    matrix = confusion_matrix(y_test.numpy(), predictions[name], labels=range(10), normalize="true")
    ConfusionMatrixDisplay(matrix, display_labels=range(10)).plot(
        ax=ax, cmap="Blues", colorbar=False, include_values=False, im_kw={"vmin": 0, "vmax": 1}
    )
    ax.set_title(f"{name}: accuracy = {results[name]['test_accuracy']:.2%}")
    ax.set_xlabel("Przewidywana cyfra")
    ax.set_ylabel("Prawdziwa cyfra")
fig.colorbar(axes[-1].images[0], ax=axes, fraction=0.02, pad=0.02, label="Udział w prawdziwej klasie")
fig.savefig(OUTPUT_DIR / "macierze_pomylek.png", bbox_inches="tight")
plt.show()
""")
md(r"""
## 7. Czym jest t-SNE i jak czytać wykresy?

t-SNE (*t-distributed Stochastic Neighbor Embedding*) zamienia wielowymiarowe wektory w punkty na płaszczyźnie. W przybliżeniu: wyznacza podobieństwa pomiędzy przykładami w oryginalnej przestrzeni, a następnie szuka układu w 2D, który zachowuje lokalne sąsiedztwa. W 2D używa rozkładu o cięższych ogonach, co pomaga rozsunąć grupy punktów. Optymalizacja jest losowa i może prowadzić do różnych układów.

Tutaj **jeden punkt odpowiada jednemu obrazowi testowemu**. Kolor oznacza prawdziwą cyfrę. Wejściem t-SNE są kolejno: 784 piksele, wektor z `Flatten` oraz aktywacje trzech warstw ukrytych. Etykiety służą tylko do pokolorowania punktów, nie do obliczania projekcji. Same cechy sieci powstały jednak dzięki uczeniu z etykietami.

`perplexity=30` określa skalę rozpatrywanego sąsiedztwa; nie jest liczbą klas. Musi być mniejsza od liczby punktów. Dla wektorów o ponad 50 współrzędnych używamy najpierw PCA do 50 wymiarów, co przyspiesza obliczenia. PCA może odrzucić część informacji, dlatego t-SNE obrazuje tu cechy po tej redukcji. Wyższe warstwy mają już 64 lub 32 cechy; dla 64 też stosujemy PCA, dla 32 nie.

**Ograniczenia interpretacji:**

- Lokalnie mieszające się kolory mogą wskazywać na słabą separację klas w danej reprezentacji.
- Odległości pomiędzy odległymi grupami, ich powierzchnie i orientacje nie są wiarygodną miarą jakości klasyfikatora.
- Każdy panel to osobno obliczone t-SNE; osie różnych paneli nie mają wspólnego znaczenia.
- Wizualizacja nie zastępuje accuracy testowej. Nie wyliczamy accuracy z punktów 2D.
- Jedno ziarno i jedna wartość perplexity dają ilustrację. Mocniejsze wnioski wymagają powtórzeń.

Zobacz [dokumentację `sklearn.manifold.TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
""")
code(r"""
if not 1 < TSNE_SAMPLES < len(y_test):
    raise ValueError("TSNE_SAMPLES musi być pomiędzy 2 a liczbą obrazów testowych minus 1.")
if not 0 < TSNE_PERPLEXITY < TSNE_SAMPLES:
    raise ValueError("Perplexity musi być dodatnie i mniejsze od TSNE_SAMPLES.")
tsne_ids, _ = train_test_split(
    np.arange(len(y_test)), train_size=TSNE_SAMPLES,
    stratify=y_test.numpy(), random_state=SEED
)
tsne_labels = y_test.numpy()[tsne_ids]
np.save(OUTPUT_DIR / "indeksy_tsne.npy", tsne_ids)

@torch.inference_mode()
def extract_representations(model, images):
    model.eval()
    collected = {name: [] for name in ("Piksele", "Flatten", "MLP 1", "MLP 2", "MLP 3")}
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE].to(DEVICE, dtype=torch.float32).div(255.0)
        collected["Piksele"].append(batch.flatten(1).cpu().numpy())
        for name, tensor in model.representations(batch).items():
            collected[name].append(tensor.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}

def project_tsne(features):
    pca_dimensions = min(50, features.shape[1], features.shape[0] - 1)
    if features.shape[1] > pca_dimensions:
        features = PCA(n_components=pca_dimensions, random_state=SEED).fit_transform(features)
    settings = dict(n_components=2, perplexity=TSNE_PERPLEXITY,
                    init="pca", learning_rate="auto", random_state=SEED)
    # W sklearn >=1.5 parametr nazywa się max_iter; starsze wersje używały n_iter.
    iteration_parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    settings[iteration_parameter] = TSNE_ITERATIONS
    return TSNE(**settings).fit_transform(features)

representations, embeddings = {}, {}
for name in ("A", "B", "C"):
    representations[name] = extract_representations(models[name], variants[name]["test"][tsne_ids])
    embeddings[name] = {}
    for layer, features in representations[name].items():
        print(f"t-SNE: wariant {name}, {layer}, wejście {features.shape}", flush=True)
        embeddings[name][layer] = project_tsne(features)
    np.savez(OUTPUT_DIR / f"tsne_{name}.npz", **embeddings[name])
print("Projekcje zakończone.")
""")
code(r"""
for name in ("A", "B", "C"):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7), layout="constrained")
    for ax, (layer, points) in zip(axes, embeddings[name].items()):
        scatter = ax.scatter(points[:, 0], points[:, 1], c=tsne_labels,
                             cmap="tab10", vmin=-0.5, vmax=9.5, s=7, alpha=0.75, linewidths=0)
        ax.set_title(f"{layer}\n{representations[name][layer].shape[1]} cech przed PCA")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(scatter, ax=axes, ticks=range(10), label="Prawdziwa cyfra", shrink=0.85, pad=0.015)
    fig.suptitle(f"Wariant {name}: reprezentacje tych samych {TSNE_SAMPLES} obrazów testowych")
    fig.savefig(OUTPUT_DIR / f"tsne_{name}.png", bbox_inches="tight")
    plt.show()

# Kontrola ważnej własności wariantu B: wspólna permutacja zachowuje odległości euklidesowe.
original_vectors = representations["A"]["Piksele"]
permuted_vectors = representations["B"]["Piksele"]
distance_A = np.linalg.norm(original_vectors[1:50] - original_vectors[0], axis=1)
distance_B = np.linalg.norm(permuted_vectors[1:50] - permuted_vectors[0], axis=1)
assert np.allclose(distance_A, distance_B, atol=1e-5)
print("B zachowuje odległości między surowymi wektorami pikseli (kontrola numeryczna).")
""")
md(r"""
## 8. Wyjaśnienie oczekiwanych różnic

**A — oryginalne obrazy.** Sąsiednie piksele tworzą fragmenty tej samej kreski. To odpowiada założeniu konwolucji: lokalny filtr wykrywa powtarzalne wzorce. Przewidujemy wysoką skuteczność i coraz wyraźniejsze rozdzielanie klas w reprezentacjach MLP.

**B — wspólna permutacja.** Znika naturalna lokalność obrazu, ale przekształcenie jest stałe i odwracalne. Sama permutacja nie usuwa informacji. Dla wektorów $x$ i $y$ oraz wspólnej macierzy permutacji $P$ zachodzi $\|Px-Py\|_2=\|x-y\|_2$. Zatem podobieństwa między surowymi obrazami A i B są takie same w metryce euklidesowej. Wykresy t-SNE wejścia mogą być podobne, choć niezależne obliczenia PCA/t-SNE nie gwarantują identycznej orientacji ani układu.

Różnica w treningu CNN wynika z tego, że filtr 3 × 3 i pooling działają teraz na pikselach, które wcześniej mogły być daleko od siebie. Pooling może też usuwać przydatne informacje. MLP może mimo tego wykorzystać stałą organizację wejścia. Przewidujemy zwykle gorszy wynik niż A przy krótkim treningu, ale zdecydowanie lepszy niż losowe zgadywanie. MLP działający bezpośrednio na pikselach mógłby teoretycznie skompensować wspólną permutację zmianą wag swojej pierwszej warstwy — tego dodatkowego modelu tutaj nie trenujemy.

**C — osobne permutacje.** Ta sama pozycja nie ma stałego znaczenia pomiędzy przykładami. Model nie może nauczyć się jednej odwrotnej permutacji dla wszystkich obrazów. W ten sposób znika spójna informacja o kształcie, lecz zachowany jest histogram jasności, ilość jasnych pikseli i inne statystyki niezależne od kolejności. Cyfry mogą różnić się tymi statystykami, więc **wynik C nie musi wynosić 10%**. 10% to punkt odniesienia dla równomiernego losowania jednej z dziesięciu klas. Faktyczna jakość zależy od danych, architektury i treningu.

Wnioski o przyczynach mają charakter interpretacji doświadczenia. Jeden przebieg nie dowodzi uniwersalnego porządku skuteczności A > B > C dla każdej sieci i każdego ustawienia.
""")
md(r"""
### Obserwacje z zapisanego uruchomienia

Poniższy opis dotyczy wykresów zapisanych w tym notebooku: ziarno 42, 20 000 obrazów treningowych i 5 epok. Po zmianie konfiguracji należy porównać go z nowymi wynikami. Dokładne wartości accuracy znajdują się w tabeli i automatycznym podsumowaniu poniżej.

**A:** po pierwszej epoce accuracy walidacyjna przekracza 87%, a końcowy wynik testowy przekracza 96%. W macierzy pomyłek wyraźnie dominuje przekątna. Na t-SNE surowych pikseli część kolorów się miesza; po `Flatten` grupy są czytelniejsze, a w `MLP 1–3` większość cyfr tworzy wyraźne grupy o jednym dominującym kolorze. To jest zgodne z uczeniem reprezentacji przydatnej do klasyfikacji.

**B:** wynik testowy wynosi około 90%, czyli model nauczył się dużo pomimo zniszczenia lokalnego kształtu. Panele `Piksele` dla A i B wyglądają niemal identycznie, co odpowiada zachowaniu odległości przez wspólną permutację. Reprezentacja `Flatten` B ma więcej mieszających się kolorów niż A. W kolejnych warstwach MLP pojawiają się wyraźniejsze grupy, ale nadal np. cyfry 3, 5 i 8 częściowo zajmują wspólny obszar. To jakościowo odpowiada większej liczbie pomyłek niż w A.

**C:** accuracy treningowa i walidacyjna utrzymują się blisko 25%, a wynik testowy jest wyższy od losowych 10%, lecz znacznie niższy niż A i B. Podobnie niskie wyniki treningu i walidacji wskazują, że w tym krótkim przebiegu główną trudnością jest nauczenie skutecznej reguły, a nie duża luka między treningiem i walidacją. W macierzy pomyłek cyfra 1 jest rozpoznawana wyraźnie lepiej od większości pozostałych cyfr. Możliwe wyjaśnienie to jej charakterystyczna ilość jasnych pikseli; doświadczenie nie izoluje jednak tej przyczyny.

Na t-SNE C kolory w większości się mieszają. W MLP powstają łukowate pasma, ale pasmo samo w sobie nie oznacza klasy — większość pasm zawiera wiele kolorów. Punkty cyfry 1 tworzą bardziej odrębną część. Dlatego oceniając t-SNE, trzeba patrzeć na etykiety punktów oraz wyniki klasyfikacji, a nie tylko na atrakcyjny kształt projekcji.
""")
code(r'''
# Podsumowanie tworzymy z faktycznie zmierzonych wyników, bez wpisywania oczekiwanych liczb.
ranking = sorted(results, key=lambda name: results[name]["test_accuracy"], reverse=True)
accuracy_A, accuracy_B, accuracy_C = [results[name]["test_accuracy"] for name in ("A", "B", "C")]
random_baseline = 0.1
majority_baseline = np.bincount(y_test.numpy(), minlength=10).max() / len(y_test)
report = f"""### Podsumowanie wykonanego eksperymentu

- Użyto {len(train_ids)} przykładów treningowych, {len(val_ids)} walidacyjnych i {len(y_test)} testowych; wykonano {EPOCHS} epok na wariant.
- Accuracy testowa: **A = {accuracy_A:.2%}, B = {accuracy_B:.2%}, C = {accuracy_C:.2%}**.
- Ranking w tym uruchomieniu: **{' > '.join(ranking)}** (ewentualne remisy rozstrzyga kolejność wariantów).
- Różnica A minus B: {(accuracy_A - accuracy_B) * 100:+.2f} punktów procentowych; A minus C: {(accuracy_A - accuracy_C) * 100:+.2f} pp.
- Równomierne zgadywanie daje średnio {random_baseline:.0%}; przewidywanie najczęstszej klasy testowej daje {majority_baseline:.2%} (opisowy punkt odniesienia).
- Wybrano epoki na podstawie straty walidacyjnej, bez dostrajania do accuracy testowej.

Na wykresach t-SNE sprawdź, w których warstwach punkty o tym samym kolorze tworzą lokalne grupy. W opisie raportu odwołaj się do konkretnych paneli, a jako miarę skuteczności podaj accuracy i macierze pomyłek. Wyniki dotyczą pojedynczego ziarna i ograniczonego czasu uczenia.
"""
display(Markdown(report))
(OUTPUT_DIR / "podsumowanie.md").write_text(report, encoding="utf-8")

with (OUTPUT_DIR / "wyniki.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["variant", "test_accuracy", "test_loss", "best_epoch", "seconds"])
    writer.writeheader()
    for name, result in results.items():
        writer.writerow({"variant": name, **result})
configuration = {
    "seed": SEED, "seed_permutation_B": SEED + 1,
    "seed_permutation_C_train": SEED + 2, "seed_permutation_C_test": SEED + 3,
    "train_samples": len(train_ids), "validation_samples": len(val_ids), "test_samples": len(y_test),
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
    "tsne_samples": TSNE_SAMPLES, "tsne_perplexity": TSNE_PERPLEXITY, "tsne_iterations": TSNE_ITERATIONS,
    "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
    "python": platform.python_version(), "device": str(DEVICE),
}
(OUTPUT_DIR / "konfiguracja.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
(OUTPUT_DIR / "historia.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
print("Zapisano modele, wykresy, podziały danych, projekcje t-SNE i wyniki do:", OUTPUT_DIR.resolve())
''')
md(r"""
## 9. Co dalej na zajęciach?

Notebook jest gotowy do dopisania dalszych celów. Jeśli prowadzący poprosi o dłuższy trening, zmień `EPOCHS`. Aby użyć całej dostępnej części treningowej, ustaw `N_TRAIN = None`. Po zmianie ustawień uruchom notebook od początku, aby wszystkie warianty miały tę samą konfigurację.

Przy rozbudowie doświadczenia warto sprawdzić kilka ziaren, kilka wartości perplexity i porównać CNN z MLP uczonym bezpośrednio na pikselach. Rozróżniaj dwa doświadczenia: **uczenie i test na tym samym wariancie** (wykonane tutaj) oraz uczenie na A i test na B/C (zmiana rozkładu wejścia, osobne zagadnienie).

Materiały źródłowe:

- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — funkcja straty dla logitów.
- [Torchvision: MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) — pobieranie danych.
- [scikit-learn: t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) — parametry i ograniczenia redukcji wymiarów.

Jeśli oddajesz notebook jako własny raport, uzupełnij wymagane dane autora i własne obserwacje z wykresów oraz informację o użyciu AI zgodnie z zasadami zajęć.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12", "file_extension": ".py", "mimetype": "text/x-python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(cells):
    cell["id"] = f"lab2-{index:02d}"
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lab2.ipynb")
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Zapisano {output_path}: {len(cells)} komórek.")
