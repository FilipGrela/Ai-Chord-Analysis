# Surowy Zbiór Danych (Raw Dataset)

Ten folder jest przeznaczony na pliki audio oraz etykiety akordów (ang. labels) niezbędne do wygenerowania macierzy uczących dla modelu. 

Ze względu na rozmiar plików audio (.mp3, .wav) oraz prawa autorskie, **zawartość tego folderu jest ignorowana przez system Git** (nie trafia na GitHuba).

---

##  Wymagana struktura katalogów

Aby skrypt budujący dataset poprawnie odczytał utwory, każdy z nich musi znajdować się w **osobnym podfolderze**. Wewnątrz podfolderu muszą być dokładnie dwa pliki: jedno audio i jedna etykieta.

Przykład poprawnej struktury:
- album_1/
  - utwor_01/
    - dzwiek.mp3
    - etykiety.jams
  - utwor_02/
    - audio.wav
    - chords.csv
- album 2
  - ...

---

## Obsługiwane formaty plików

### 1. Pliki Audio
* `.mp3`
* `.wav`
*(Wymagany zainstalowany w systemie FFmpeg do poprawnego odczytu przez bibliotekę procesującą).*

### 2. Pliki Etykiet (Labels)
System obsługuje 3 formaty etykiet. Parser automatycznie znormalizuje nazwy akordów (np. zamieni Db na C#, usunie inwersje).

* **Format .jams** (Zalecany): Pliki JSON ustandaryzowane dla analizy muzycznej (wymagany namespace `chord`).
* **Format .csv lub .txt**: Proste pliki tekstowe. Separator to przecinek lub tabulacja. Brak nagłówków.

**Wymagany format wiersza w plikach CSV/TXT:**
`[czas_rozpoczęcia_w_sekundach], [czas_zakończenia_w_sekundach], [nazwa_akordu]`

**Przykład poprawnego pliku .csv:**
0.000, 1.250, C
1.250, 2.800, G
2.800, 4.100, Am
4.100, 5.000, N

*(Uwaga: N oznacza brak akordu / ciszę / samą perkusję).*