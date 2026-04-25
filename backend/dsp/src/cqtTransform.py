import numpy as np
import librosa

class CqtTransform:
    def __init__(self, binsPerOctave: int, fMin: float, fS: int, hopLength: int, nBins: int = 84):
        self.binsPerOctave = binsPerOctave
        self.fMin = fMin
        self.fS = fS
        self.hopLength = hopLength
        self.nBins = nBins
        self.Q = (1 / (2 ** (1 / self.binsPerOctave) - 1)) * 2.0

        self.kernels = []
        self.generateKernels()

    def generateKernels(self):
        self.kernels = []
        for k in range(self.nBins):
            fK = self.fMin * 2 ** (k / self.binsPerOctave)
            nK = int(np.ceil(self.Q * self.fS / fK))

            window = np.blackman(nK)
            kernel = np.exp(-2j * np.pi * fK * np.arange(nK) / self.fS)
            kernel = (kernel * window) / nK
            
            self.kernels.append(kernel)

    def processAudio(self, audioData):
        results = []

        maxKernelLen = len(self.kernels[0])
        if(len(audioData) < maxKernelLen):
            audioData = np.pad(audioData, (0, maxKernelLen - len(audioData)), mode='constant')

        # Stala liczba ramek dla wszystkich binow (wg najdluzszego kernela),
        # aby wynik zawsze mial ksztalt (nBins, frames).
        targetFrames = 1 + (len(audioData) - maxKernelLen) // self.hopLength

        # Cache ramek dla powtarzajacych sie dlugosci kernela.
        framedCache = {}
    
        for k in range(self.nBins):
            kernel = self.kernels[k]
            nK = len(kernel)

            if nK not in framedCache:
                framedCache[nK] = librosa.util.frame(audioData, frame_length=nK, hop_length=self.hopLength)

            frames = framedCache[nK]
            magnitude = np.abs(np.dot(np.conj(kernel), frames))
            magnitude = magnitude[:targetFrames]

            results.append(magnitude)
        
        return np.array(results)
    
    def toSpectrogram(self, cqtMatrix):
        cqtDb = 20 * np.log10(cqtMatrix + 1e-10)
        
        dynamicRange = 35.0
        maxDb = np.max(cqtDb)
        thresholdDb = maxDb - dynamicRange
        
        cqtDb[cqtDb < thresholdDb] = thresholdDb
        spectrogram = (cqtDb - thresholdDb) / dynamicRange
        
        return spectrogram
    
    def toChromagram(self, cqtMatrix):
        nBins, _ = cqtMatrix.shape
        targetBins = 84 
        
        paddedCqt = np.pad(cqtMatrix, ((0, targetBins - nBins), (0, 0)), mode='constant')
        
        reshaped = paddedCqt.reshape(7, 12, -1)
        chromagram = reshaped.sum(axis=0)
        
        chromagram = np.log1p(100 * chromagram)
        
        cMin = chromagram.min()
        cMax = chromagram.max()
        chromagram = (chromagram - cMin) / (cMax - cMin + 1e-6)

        chromagram[chromagram < 0.2] = 0

        return chromagram