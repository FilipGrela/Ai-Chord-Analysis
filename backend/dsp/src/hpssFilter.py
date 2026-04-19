import numpy as np
import librosa

class HpssFilter:
      def __init__(self, harmonicMargin: float, percussiveMargin: float):
            self.__harmonicMargin = harmonicMargin
            self.__percussiveMargin = percussiveMargin
            self.__y = None
            self.__sr = None

      def loadAudio(self, audioPath: str):
          self.__y, self.__sr = librosa.load(audioPath, sr=None, mono=True)

      def loadAudioArray(self, y: np.ndarray, sr: int):
          self.__y = y
          self.__sr = sr

      def extractHarmonic(self):
          if self.__y is None:
                print("Load audio first!!!")
                return None, None
          yHarmonic, _ = librosa.effects.hpss(self.__y, margin=(self.__harmonicMargin, self.__percussiveMargin))
          return yHarmonic, self.__sr #czestotliwosc probkowania potrzebna
