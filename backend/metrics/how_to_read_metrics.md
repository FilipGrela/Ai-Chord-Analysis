# Przewodnik po metrykach: AI-Chord-Analysis

Ten dokument szczegółowo opisuje metryki wykorzystywane podczas treningu i ewaluacji modelu. Każda sekcja wyjaśnia sposób obliczania danej metryki, jej matematyczne podstawy oraz praktyczne znaczenie przy analizowaniu wyników.

---

## 1. Architektura Ewaluacji

Proces monitorowania modelu dzieli się na dwa główne etapy:

*   **Trening (`backend/training/trainer.py`):** Zapisuje bieżące metryki epokowe do pliku `out/metrics/history.csv` oraz generuje wykres trendów `out/metrics/history.png`.
*   **Ewaluacja (`backend/metrics/evaluator.py`):** Zwraca kompleksowy słownik z wynikami klasyfikacji i generuje wykresy analityczne (macierz pomyłek, raport per-class, podsumowanie metryk muzycznych).

---

## 2. Funkcje Straty (Loss Metrics)

### `train_loss` / `val_loss`
Średnia wartość błędu na paczkach danych (batchach) w danej epoce:

$$
L_{epoch} = \frac{1}{N}\sum_{i=1}^{N} L_i
$$

gdzie $L_i$ to błąd dla pojedynczej paczki, a $N$ to liczba paczek.
*   **Znaczenie:** Spadek oznacza lepsze dopasowanie modelu. Wzrost na zbiorze walidacyjnym przy jednoczesnym spadku na treningowym to wyraźny sygnał **przeuczenia (overfittingu)**. Duże wahania (szum) zwykle sugerują zbyt wysoki *Learning Rate* lub niestabilne dane.

### `train_ce_hard` / `val_ce_hard`
Składowa twardej klasyfikacji (CrossEntropy) z `SoftLabelLoss`:

$$
L_{CE} = -\log p(y_{true})
$$

*   **Znaczenie:** Mierzy, jak dobrze model trafia w poprawną, docelową klasę. To Twój główny sygnał jakości twardej klasyfikacji.

### `train_kl_soft` / `val_kl_soft`
Składowa rozbieżności (KL Divergence) między matrycą podobieństwa klas a predykcją modelu:

$$
L_{KL} = \sum_j q_j (\log q_j - \log p_j)
$$

gdzie $q_j$ to miękki cel wygładzony temperaturą, a $p_j$ to przewidywane prawdopodobieństwo.
*   **Znaczenie:** Mierzy zgodność przewidywań z relacjami harmonicznymi. Niska wartość oznacza, że model trafnie rozkłada niepewność (np. wahając się między Cmaj a Gmaj, a nie Cmaj a F#m). 

### Całkowity Błąd (`loss`)
Wzór, na podstawie którego optymalizator aktualizuje wagi:

$$
L = (1-\alpha)L_{CE} + \alpha L_{KL}
$$

*   **Znaczenie:** Zbalansowany kompromis ustalany przez hiperparametr $\alpha$, łączący surową celność z muzyczną elastycznością.

---

## 3. Metryki Klasyfikacyjne

### Całkowita Dokładność (`accuracy`)
$$
\text{Accuracy} = \frac{\text{Poprawne predykcje}}{\text{Liczba wszystkich przykładów}}
$$
*   **Znaczenie:** Mierzy ogólną trafność. Przy mocno niezbalansowanych danych ulega zniekształceniu (faworyzuje najczęstsze klasy).

### Raport per-class (`per_class`)
Słownik zawierający szczegółowe dane dla każdej zidentyfikowanej klasy:

*   **Precision (Precyzja):** $\frac{TP}{TP + FP}$
    Mówi, ile z predykcji danej klasy było faktycznie poprawnych. Niska wartość to dużo fałszywych alarmów (model widzi dany akord wszędzie).
*   **Recall (Czułość):** $\frac{TP}{TP + FN}$
    Mówi, ile rzeczywistych wystąpień danej klasy model zdołał wyłapać. Niska wartość to pomijanie akordu na nagraniach.
*   **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
    Harmoniczny kompromis między precyzją a czułością.
*   **Support:** Liczba rzeczywistych próbek danej klasy w zbiorze. Wskazuje, jak bardzo miarodajny jest wynik F1.

### Metryki Uśrednione F1
*   **`macro_f1`:** Średnia arytmetyczna F1 dla wszystkich klas. Niska wartość przy wysokim `accuracy` dowodzi, że model radzi sobie dobrze tylko z popularnymi akordami.
*   **`weighted_f1`:** Średnia F1 ważona ilością wystąpień danej klasy (`support`). Lepiej odzwierciedla globalną wydajność na niezbalansowanym datasecie.

---

## 4. Metryki Muzyczne

Są to unikalne metryki domenowe, bazujące na metodzie `_split_chord()`, ignorujące sztywne etykiety na rzecz zrozumienia harmonii.

*   **`root_accuracy`:** Porównuje wyłącznie rdzeń (fundament) akordu (np. wyciąga 'C' z 'Cm7'). Sukces tutaj przy jednoczesnym błędzie `quality` oznacza, że model zlokalizował poprawną tonikę, ale pomylił tryb.
*   **`quality_accuracy`:** Porównuje wyłącznie tryb/rozszerzenie ('maj', 'm', '7', 'm7'). 
*   **`cer` (Chord Error Rate):** Najważniejsza metryka sekwencyjna. Oblicza odległość Levenshteina na skompresowanych blokach akordów:
    $$
    CER = \frac{d_{Lev}(y_{ref}, y_{pred})}{\max(1, |y_{ref}|)}
    $$
    Bada stabilność przebiegu akordów w czasie. Wysoki CER (przy niezłej dokładności bazowej) obnaża gubienie ciągłości sekwencji muzycznej (tzw. migotanie).
*   **`top_k`:** Mierzy, czy poprawna klasa znalazła się w ścisłej czołówce predykcji (np. top-3). Duża dysproporcja między `top_1` a `top_3` sugeruje, że model rozpoznaje akordy, ale brakuje mu ostatecznej pewności siebie.

---

## 5. Wykresy i Wizualizacje

*   **`history.png`:** Linie trendów funkcji strat i dokładności w czasie. Rozchodzące się nożyce między zbiorem treningowym a walidacyjnym to jednoznaczny sygnał przeuczenia.
*   **`cm_*.png` (Macierz Pomyłek):** Diagonala to sukcesy. Klastry poza przekątną pomagają zlokalizować, które akordy najczęściej mylą się ze sobą nawzajem.
*   **`per_class_*.png`:** Słupkowy profil precyzji/czułości/F1. Szybko demaskuje najgorzej radzące sobie klasy z ogona (tail classes).
*   **`metrics_summary.png`:** Mini-raport jakości muzycznej i klasyfikacyjnej. Rysuje znormalizowany wynik `1 - CER`, by wyższe słupki zawsze oznaczały lepszy rezultat.

---

## 6. Diagnozowanie Problemów (Troubleshooting)

Jak czytać anomalie i wyciągać z nich wnioski naprawcze:

1.  **Klasyczny Overfitting:** `train_loss` spada, `val_loss` rośnie.
    *   *Akcja:* Zwiększ parametr Dropout, dodaj silniejszą augmentację lub zastosuj Early Stopping.
2.  **Zaniedbane Mniejszości:** `accuracy` jest wysokie, ale `macro_f1` niskie.
    *   *Akcja:* Model faworyzuje klasy dominujące. Sprawdź poprawność wdrożenia wag klas (`Label Smoothing` / LossFactory).
3.  **Gubienie Jakości (Modu):** `root_accuracy` wysoce przekracza `quality_accuracy`.
    *   *Akcja:* Model znakomicie łapie fundament, ale ignoruje interwały. Problem może wynikać ze zbytniego wygładzania nałożonego przez matrycę podobieństw.
4.  **Brak Ciągłości:** Duży błąd `cer` mimo bardzo wysokiego `accuracy`.
    *   *Akcja:* Model trafnie ocenia pojedyncze ramki, ale migocze. Wymaga dłuższego okna czasowego (np. szerszego jądra CNN lub powiększenia RNN_HIDDEN_SIZE).
5.  **Model jest blisko, ale pudłuje:** Wartość `top_3` jest drastycznie wyższa niż `top_1`.
    *   *Akcja:* Architektura jest odpowiednia, ale proces uczenia zatrzymał się przedwcześnie. Prawdopodobnie wymaga fine-tuningu przy niższym Learning Rate.

---

## 7. Zwracany Obiekt API (`evaluate`)

Wywołanie `MetricsEvaluator.evaluate(...)` ujednolica wyniki w czytelny słownik JSON-like, gotowy do logowania na platformach MLOps (np. MLflow, W&B):

```python
{
    "accuracy": 0.845,
    "macro_f1": 0.712,
    "weighted_f1": 0.831,
    "root_accuracy": 0.910,
    "quality_accuracy": 0.765,
    "cer": 0.182,
    "top_k": {
        "top_1": 0.845, 
        "top_3": 0.961
    },
    "timestamp": "2026-05-23T15:47:48.123456"
}