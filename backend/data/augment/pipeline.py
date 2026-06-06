from typing import List, Any
import torch

from backend.data.augment.transforms import RandomGain
from backend.data.augment.transforms import RandomSpecMask
from backend.data.augment.transforms import AdditiveGaussianNoise
from backend.data.augment.transforms import RandomTranspose
from backend.logger import logger as log


class AugmentPipeline:
    def __init__(self, transforms: List[Any]):
        self.logger = log.Logger(__name__)
        self.transforms = transforms  # type: List[Any]

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec
        try:
            self.logger.info(f"Uruchamiam pipeline augmentacji z {len(self.transforms)} transformacjami")
        except Exception:
            pass
        for t in self.transforms:
            # wszystkie transformacje w tym pipeline używają interfejsu .apply(...)
            out = t.apply(out)
        return out



    # Augumentacja online pozwala edytować pisenki przy każdej epoce treningu.
    # Możliwe jest że piosenka przy 1 epoce bedzie zmodyfikowana, a przy kolejnej nie.
    # Pozwala to modelowi dopasować się do zmiennych warunków. Jednocześnie nie zwiększając danyc na dysku.
def build_train_augment_pipeline(train_cfg) -> AugmentPipeline | None:
    logger = log.Logger(__name__)
    if not getattr(train_cfg, "AUGMENT_ENABLED", False):
        logger.warning("Augmentacja wyłączona.")
        return None

    transforms: List[Any] = []
    if getattr(train_cfg, "AUGMENT_GAIN_ENABLED", False):
        logger.info("Włączono augmentację: RandomGain z parametrami: ")
        logger.info(f"  - Prob: {train_cfg.AUGMENT_GAIN_PROB}")
        logger.info(f"  - Gain dB Min: {train_cfg.AUGMENT_GAIN_DB_MIN}")
        logger.info(f"  - Gain dB Max: {train_cfg.AUGMENT_GAIN_DB_MAX}")
        transforms.append(RandomGain())

    if getattr(train_cfg, "AUGMENT_SPECMASK_ENABLED", False):
        logger.info("Włączono augmentację: RandomSpecMask z parametrami: ")
        logger.info(f"  - Prob: {train_cfg.AUGMENT_SPECMASK_PROB}")
        logger.info(f"  - Max Time Masks: {train_cfg.AUGMENT_SPECMASK_MAX_TIME_MASKS}")
        logger.info(f"  - Max Freq Masks: {train_cfg.AUGMENT_SPECMASK_MAX_FREQ_MASKS}")
        logger.info(f"  - Max Time Width: {train_cfg.AUGMENT_SPECMASK_MAX_TIME_WIDTH}")
        logger.info(f"  - Max Freq Width: {train_cfg.AUGMENT_SPECMASK_MAX_FREQ_WIDTH}")
        transforms.append(RandomSpecMask())

    if getattr(train_cfg, "AUGMENT_NOISE_ENABLED", False):
        logger.info("Włączono augmentację: AdditiveGaussianNoise z parametrami: ")
        logger.info(f"  - Prob: {train_cfg.AUGMENT_NOISE_PROB}")
        logger.info(f"  - SNR dB Min: {train_cfg.AUGMENT_NOISE_SNR_DB_MIN}")
        logger.info(f"  - SNR dB Max: {train_cfg.AUGMENT_NOISE_SNR_DB_MAX}")
        transforms.append(AdditiveGaussianNoise())

    if getattr(train_cfg, "AUGMENT_TRANSPOSE_ENABLED", False):
        logger.info("Włączono augmentację: RandomTranspose z parametrami: ")
        logger.info(f"  - Prob: {train_cfg.AUGMENT_TRANSPOSE_PROB}")
        logger.info(f"  - Min Semitones: {train_cfg.AUGMENT_TRANSPOSE_MIN}")
        logger.info(f"  - Max Semitones: {train_cfg.AUGMENT_TRANSPOSE_MAX}")
        transforms.append(RandomTranspose())

    if not transforms:
        return None

    return AugmentPipeline(transforms)

