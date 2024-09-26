from enum import Enum


class SAMModels(str, Enum):
    Tiny = "tiny"
    Small = "small"
    BasePlus = "base_plus"
    Large = "large"

    def get_config(self):
        match self:
            case SAMModels.Tiny:
                return "sam2_hiera_t.yaml"
            case SAMModels.Small:
                return "sam2_hiera_s.yaml"
            case SAMModels.BasePlus:
                return "sam2_hiera_b+.yaml"
            case SAMModels.Large:
                return "sam2_hiera_l.yaml"

    def get_checkpoint(self):
        match self:
            case SAMModels.Tiny:
                return "./checkpoints/sam2_hiera_tiny.pt"
            case SAMModels.Small:
                return "./checkpoints/sam2_hiera_small.pt"
            case SAMModels.BasePlus:
                return "./checkpoints/sam2_hiera_base_plus.pt"
            case SAMModels.Large:
                return "./checkpoints/sam2_hiera_large.pt"
