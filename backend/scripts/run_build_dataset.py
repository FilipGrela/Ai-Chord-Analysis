from backend.data.builder import DatasetBuilder
from backend.logger.logger import Logger

logger = Logger(__name__)

def main():
    logger.info("--- Inicjalizacja Pipeline'u ETL dla AI-Chord-Analysis ---")
    builder = DatasetBuilder()
    builder.build_entire_dataset()

if __name__ == "__main__":
    main()