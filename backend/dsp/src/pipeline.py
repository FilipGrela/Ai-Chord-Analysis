import numpy as np
from typing import Optional
from .hpssFilterOptimized import HpssFilter
from .cqtTransform import CqtTransform
from backend.config import AudioConfig

class Pipeline:
    def __init__(self, harmonicMargin: float, percussiveMargin: float, fMin: float, fS: int, hopLength: int, binsPerOctave: Optional[int] = None, nBins: Optional[int] = None):
        self.__hpss = HpssFilter(harmonicMargin, percussiveMargin)
        if binsPerOctave is None:
            binsPerOctave = AudioConfig.BINS_PER_OCTAVE
        if nBins is None:
            nBins = AudioConfig.N_BINS
        self.__cqt = CqtTransform(binsPerOctave, fMin, fS, hopLength, nBins=nBins)

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

        if chroma is None:
            raise ValueError("Chroma data is None. Ensure audio was loaded and processed correctly.")

        chromaColumns = chroma.shape[1]

        chunkNr = chromaColumns // frames
        chunks = []

        for i in range(chunkNr):
            begin = i * frames
            end = begin + frames

            chunk = chroma[:, begin:end].copy()
            chunks.append(chunk)

        return np.stack(chunks)
