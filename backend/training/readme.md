# SILNIK UCZENIA (TRAINING LOOP)
Folder zawiera logikę optymalizacji modelu i nadzoru nad treningiem.

- trainer.py: Klasa 'Trainer'. Zarządza pętlą 'model.train()', walidacją, zapisywaniem najlepszych wag (Checkpointing) oraz mechanizmem 'EarlyStopping' (przerywa trening, gdy model przestaje się uczyć).
- loss.py: 'LossFactory'. Oblicza wagi dla CrossEntropyLoss na podstawie pierwiastka z liczności klas, co zapobiega ignorowaniu rzadkich akordów przez model.