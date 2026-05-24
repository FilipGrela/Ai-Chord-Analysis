import torch
import numpy as np
import librosa
from backend.logger.logger import Logger

logger = Logger(__name__)

class HpssFilter:
    def __init__(self, harmonicMargin: float, percussiveMargin: float, 
                 kernelSize: int = 31, nFft: int = 2048, hopLength: int = 512):
        self.__harmonicMargin = harmonicMargin
        self.__percussiveMargin = percussiveMargin
        self.__kernelSize = kernelSize
        self.__nFft = nFft
        self.__hopLength = hopLength
        self.__y = None
        self.__sr = None
        
        # Automatyczne wykorzystanie rdzeni CUDA
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__window = torch.hann_window(nFft).to(self.__device)

    def loadAudio(self, audioPath: str):
        self.__y, self.__sr = librosa.load(audioPath, sr=None, mono=True)

    def loadAudioArray(self, y: np.ndarray, sr: int):
        self.__y = y
        self.__sr = sr

    def extractHarmonic(self):
        if self.__y is None:
            logger.error("Load audio first!!!")
            return None, None
            
        y_tensor = torch.tensor(self.__y, dtype=torch.float32, device=self.__device)
        
        stft_complex = torch.stft(
            y_tensor,
            n_fft=self.__nFft,
            hop_length=self.__hopLength,
            window=self.__window,
            return_complex=True,
            pad_mode='reflect'
        )
        magnitude = torch.abs(stft_complex)
        
        # --- POPRAWKA: Dodajemy wymiar batcha, aby tensor był 3D ---
        magnitude = magnitude.unsqueeze(0) # Teraz: [1, Freq, Time]
        
        pad_size = self.__kernelSize // 2
        
        # Filtracja harmoniczna wzdłuż osi czasu (wymiar 2)
        mag_padded_t = torch.nn.functional.pad(magnitude, (pad_size, pad_size, 0, 0), mode='reflect')
        S_h = mag_padded_t.unfold(-1, self.__kernelSize, 1).median(dim=-1).values
        
        # Filtracja perkusyjna wzdłuż osi częstotliwości (wymiar 1)
        mag_padded_f = torch.nn.functional.pad(magnitude, (0, 0, pad_size, pad_size), mode='reflect')
        S_p = mag_padded_f.unfold(-2, self.__kernelSize, 1).median(dim=-1).values
        
        # --- POPRAWKA: Usuwamy wymiar batcha ---
        magnitude = magnitude.squeeze(0)
        S_h = S_h.squeeze(0)
        S_p = S_p.squeeze(0)
        
        # 4. Aplikacja marginesów separacji
        mask_h = S_h > (S_p * self.__harmonicMargin)
        
        # 5. Nałożenie twardej maski
        stft_harm = stft_complex * mask_h.to(stft_complex.dtype)
        
        y_harm = torch.istft(
            stft_harm,
            n_fft=self.__nFft,
            hop_length=self.__hopLength,
            window=self.__window,
            length=y_tensor.size(0)
        )
        
        return y_harm.cpu().numpy(), self.__sr