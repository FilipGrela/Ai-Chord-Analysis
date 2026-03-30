from backend.data.builder import DatasetBuilder

def main():
    print("--- Inicjalizacja Pipeline'u ETL dla AI-Chord-Analysis ---")
    builder = DatasetBuilder()
    builder.build_entire_dataset()

if __name__ == "__main__":
    main()