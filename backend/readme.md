# BACKEND - MAPA MODUŁÓW

Ten folder zawiera cały backend projektu: konfigurację, przetwarzanie danych, DSP, architekturę modelu, trening, metryki, inferencję i skrypty uruchomieniowe.

## 1. config.py
Centralna konfiguracja projektu. Trzyma parametry dla loggera, ścieżek, audio, modelu, treningu, buildera i analizy.

## 2. logger/
Warstwa logowania aplikacji. Zawiera wspólnego loggera używanego przez wszystkie moduły backendu.

## 3. data/
Warstwa ETL i DataLoaderów.
- builder.py: budowa zbioru danych z surowego audio.
- loader.py: dataset oraz factory do DataLoaderów.
- parser.py: normalizacja etykiet akordów.
- augment/: transformacje augmentacyjne stosowane do danych treningowych.

## 4. dsp/
Moduł przetwarzania sygnału audio.
- spectrograms.py: ekstrakcja cech CQT i preprocessing audio.
- plot.py: wizualizacje spektrogramów i wykresów.
- src/: wewnętrzne elementy pipeline DSP.

## 5. models/
Definicje architektur sieci neuronowych.
- crnn.py: główny model ChordCRNN.

## 6. training/
Logika uczenia modelu.
- loss.py: budowa funkcji straty i wag klas.
- trainer.py: pętla treningowa, walidacja, checkpointing i early stopping.

## 7. metrics/
Narzędzia do ewaluacji jakości modelu i generowania wykresów metryk.

## 8. analysis/
Warstwa analityczna dla metryk muzycznych i porównań jakości danych/modeli.

## 9. api/
Warstwa inferencji. Ładowanie checkpointów i predykcja akordów dla nowych plików audio.

## 10. scripts/
Skrypty uruchomieniowe do treningu, testów, predykcji i diagnostyki.