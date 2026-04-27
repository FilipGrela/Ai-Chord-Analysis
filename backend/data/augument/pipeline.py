from typing import List
import torch

from backend.data.augument.transforms import RandomSpecMask
from backend.logger import logger as log


class AugmentPipeline:
    def __init__(self, transforms: List[object]):
        self.logger = log.Logger(__name__)
        self.transforms = transforms

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec
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

    transforms: List[object] = []
    if getattr(train_cfg, "AUGMENT_SPECMASK_ENABLED", False):
        logger.info("Włączono augmentację: RandomSpecMask z parametrami: ")
        logger.info(f"  - Prob: {train_cfg.AUGMENT_SPECMASK_PROB}")
        logger.info(f"  - Max Time Masks: {train_cfg.AUGMENT_SPECMASK_MAX_TIME_MASKS}")
        logger.info(f"  - Max Freq Masks: {train_cfg.AUGMENT_SPECMASK_MAX_FREQ_MASKS}")
        logger.info(f"  - Max Time Width: {train_cfg.AUGMENT_SPECMASK_MAX_TIME_WIDTH}")
        logger.info(f"  - Max Freq Width: {train_cfg.AUGMENT_SPECMASK_MAX_FREQ_WIDTH}")
        transforms.append(RandomSpecMask())

    if not transforms:
        return None

    return AugmentPipeline(transforms)

