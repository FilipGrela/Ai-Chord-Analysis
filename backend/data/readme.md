# WARSTWA DANYCH (ETL & PYTORCH LOADERS)
Zarządzanie cyklem życia danych – od surowego pliku po tensor w pamięci GPU.

- builder.py: 'DatasetBuilder' - serce ETL. Wykorzystuje ProcessPoolExecutor do równoległej ekstrakcji cech CQT.
- loader.py: Definiuje klasę 'ChordDataset' oraz 'DataLoaderFactory'. Odpowiada za podział danych na Train/Val (Split na poziomie utworów, nie ramek).
- parsers.py: 'ChordLabelParser' - normalizuje nazwy akordów (np. C:maj7 -> C) i mapuje je na 25 klas (12 Maj, 12 Min, 1 N).