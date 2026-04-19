import numpy as np
from .hpssFilter import HpssFilter
from .cqtTransform import CqtTransform

class Pipeline:
    def __init__(self, harmonicMargin: float, percussiveMargin: float, fMin: float, fS: int, hopLength: int):
        self.__hpss = HpssFilter(harmonicMargin, percussiveMargin)
        self.__cqt = CqtTransform(12, fMin, fS, hopLength)

    def processArrayForAI(self, audioData: np.ndarray, fS: int):
        self.__hpss.loadAudioArray(audioData, fS)
        yHarm, _ = self.__hpss.extractHarmonic()
        rawCQT = self.__cqt.processAudio(yHarm)
        
        spectrogram = self.__cqt.toSpectrogram(rawCQT)
        
        return spectrogram

    def processSound(self, audio: str):
        self.__hpss.loadAudio(str(audio))
        yHarm , _ = self.__hpss.extractHarmonic()
        rawCQT = self.__cqt.processAudio(yHarm)

        chroma = self.__cqt.toChromagram(rawCQT)

        return chroma
    
    def processSong(self, audio: str, frames: int):
        chroma = self.processSound(audio)

        chromaColumns = chroma.shape[1]

        chunkNr = chromaColumns // frames
        chunks = []

        for i in range(chunkNr):
            begin = i * frames
            end = begin + frames

            chunk = chroma[:, begin:end].copy()
            chunks.append(chunk)

        return np.stack(chunks)
