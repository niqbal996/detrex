from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_syn_v2_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
            L(T.MotionBlurAbt)(blur_limit=13, p=0.8),
            L(T.BlurAbt)(blur_limit=13,p=0.8),
            L(T.MedianBlurAbt)(blur_limit=13, p=0.8),
            L(T.ToGrayAbt)(p=0.01),
            L(T.ClaheAbt)(p=0.3),
            L(T.RandomBrightnessContrastAbt)(p=0.05),
            L(T.RandomGammaAbt)(p=0.01),
            L(T.ImageCompressionAbt)(quality_lower=75, p=0.5)
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_real_v2_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
# A.Blur(p=0.01),
# A.MedianBlur(p=0.01),
# A.ToGray(p=0.01),
# A.CLAHE(p=0.01),
# A.RandomBrightnessContrast(p=0.0),
# A.RandomGamma(p=0.0),
# A.ImageCompression(quality_lower=75, p=0.0)