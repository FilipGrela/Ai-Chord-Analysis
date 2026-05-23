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

    def processAudio(self, audioData: np.ndarray) -> np.ndarray:
        maxKernelLen = len(self.kernels[0])
        
        if len(audioData) < maxKernelLen:
            audioData = np.pad(audioData, (0, maxKernelLen - len(audioData)), mode='constant')

        targetFrames = 1 + (len(audioData) - maxKernelLen) // self.hopLength
        
        # Słownik do grupowania kerneli według ich długości
        grouped_kernels = {}
        for k in range(self.nBins):
            kernel = self.kernels[k]
            nK = len(kernel)
            if nK not in grouped_kernels:
                grouped_kernels[nK] = []
            grouped_kernels[nK].append((k, kernel))

        results = np.zeros((self.nBins, targetFrames), dtype=np.float32)

        for nK, items in grouped_kernels.items():
            indices = [item[0] for item in items]
            
            # Konwersja do complex64, aby oszczędzić 50% pamięci RAM
            kernel_matrix = np.vstack([item[1] for item in items]).astype(np.complex64)
            
            # Wyciągamy ramki audio (strided array - nie kopiuje pamięci)
            frames = librosa.util.frame(audioData, frame_length=nK, hop_length=self.hopLength)
            
            # Macierzowe mnożenie zamiast pojedynczych dot-productów
            magnitude_group = np.abs(np.matmul(np.conj(kernel_matrix), frames))
            
            for i, k_idx in enumerate(indices):
                results[k_idx, :] = magnitude_group[i, :targetFrames]

        return results
    
    def toSpectrogram(self, cqtMatrix):
        cqtDb = 20 * np.log10(cqtMatrix + 1e-10)
        
        dynamicRange = 35.0
        maxDb = np.max(cqtDb)
        thresholdDb = maxDb - dynamicRange
        
        cqtDb[cqtDb < thresholdDb] = thresholdDb
        spectrogram = (cqtDb - thresholdDb) / dynamicRange
        
        return spectrogram
    
    def toChromagram(self, cqtMatrix):
        nBins, nFrames = cqtMatrix.shape
        octaves = nBins / self.binsPerOctave
        
        if octaves != int(octaves):
            raise ValueError(f"nBins ({nBins}) must be divisible by binsPerOctave ({self.binsPerOctave})")
        
        octaves = int(octaves)
        
        # Reshape: (nBins, nFrames) -> (octaves, bins_per_octave, nFrames)
        reshaped = cqtMatrix.reshape(octaves, self.binsPerOctave, nFrames)
        
        # Sum across octaves: (octaves, bins_per_octave, nFrames) -> (bins_per_octave, nFrames)
        chromagram = reshaped.sum(axis=0)
        
        chromagram = np.log1p(100 * chromagram)
        
        cMin = chromagram.min()
        cMax = chromagram.max()
        chromagram = (chromagram - cMin) / (cMax - cMin + 1e-6)

        chromagram[chromagram < 0.2] = 0

        return chromagram