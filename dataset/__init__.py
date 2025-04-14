from .zjumocap import ZJUMoCapDataset
from .people_snapshot_IA import PeopleSnapshotDataset
from .rana_IA import RANADataset
from .synthetichuman_IA import SynthetichumanDataset
from .animation_IA import AnimationDataset
from .distill_IA import DistillDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
        'rana': RANADataset,
        'synthetichuman': SynthetichumanDataset,
        'animation': AnimationDataset,
        'distill': DistillDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
