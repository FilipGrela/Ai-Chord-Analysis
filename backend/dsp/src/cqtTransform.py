import numpy as np

class CqtTransform:
    def __init__(self, binsPerOctave: int, fMin: float, fS: int, hopLength: int):
        self.binsPerOctave = binsPerOctave
        self.fMin = fMin
        self.fS = fS
        self.hopLength = hopLength
        self.kernels = []
        self.kernelsFft = []
        self.Q = (1 / (2 ** (1 / self.binsPerOctave) - 1)) * 2.0

    def generateKernels(self, nSamples: int):
        self.kernels = []
        self.kernelsFft = []
        for k in range(84):
            fK = self.fMin * 2 ** (k / self.binsPerOctave)
            nK = int(np.ceil(self.Q * self.fS / fK))

            window = np.blackman(nK)
            kernel = np.exp(-2j * np.pi * fK * np.arange(nK) / self.fS)
            kernel = (kernel * window) / nK
            
            self.kernels.append(kernel)
            self.kernelsFft.append(np.fft.fft(np.conj(kernel), nSamples))

    def processAudio(self, audioData):
        nSamples = len(audioData)

        if not self.kernelsFft or len(self.kernelsFft[0]) != nSamples:
            self.generateKernels(nSamples)

        audioFft = np.fft.fft(audioData) 
        results = []
        
        for kFft in self.kernelsFft:       
            outputFft = audioFft * kFft
            outputTime = np.fft.ifft(outputFft)
            
            magnitudes = np.abs(outputTime[::self.hopLength])
            results.append(magnitudes)

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