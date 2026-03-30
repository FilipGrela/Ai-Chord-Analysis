# DEFINICJE ARCHITEKTUR AI
Miejsce na definicje struktur sieci neuronowych (PyTorch Modules).

- crnn.py: Implementacja ChordCRNN. 
- Struktura: 3 warstwy Conv2D (ekstrakcja cech widmowych) -> Bi-directional GRU (analiza następstwa akordów w czasie) -> Global Average Pooling -> Linear Classifier.