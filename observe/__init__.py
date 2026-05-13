"""
observe/
========
KB-augmented OOD plant-disease classifier.

Trains on Bugwood (Tomato by default), evaluates on PlantVillage and
PlantWild via PathomeDB-derived text prototypes. Architecture:

    image -> SigLIP-2 vision tower (frozen base + LoRA) -> embedding
                                                              v
    class prototype texts (canonical+regional KB blocks; synthetic
    healthy templates) -> SigLIP-2 text tower (frozen) -> [C, D]
                                                              v
    classifier:  argmax( cosine(image, class_proto) * temperature )

Open vocabulary: at test time, any disease with a KB prototype can be
scored, including diseases never seen as training images.
"""

from .dataset import (
    BugwoodTomatoDataset,
    ClassIndex,
    PVFolderDataset,
    PWFolderDataset,
)
from .inference import OBSERVEInference
from .model import OBSERVE, ClassificationResult
from .prototypes import (
    add_healthy_prototypes,
    build_disease_prototype,
    build_healthy_prototype,
    load_seed_prototypes,
)
from .trainer import (
    ImageCollator,
    OBSERVETrainer,
    encode_class_prototypes,
    split_indices,
)

__all__ = [
    # model
    "OBSERVE", "ClassificationResult",
    # data
    "BugwoodTomatoDataset", "ClassIndex",
    "PVFolderDataset", "PWFolderDataset",
    # prototypes
    "build_disease_prototype", "build_healthy_prototype",
    "load_seed_prototypes", "add_healthy_prototypes",
    # training
    "OBSERVETrainer", "ImageCollator", "encode_class_prototypes",
    "split_indices",
    # inference
    "OBSERVEInference",
]
