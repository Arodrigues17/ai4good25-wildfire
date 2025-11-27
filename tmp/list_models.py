from terratorch.registry import BACKBONE_REGISTRY

for model_name in BACKBONE_REGISTRY.get_source("terratorch").keys():
    print(model_name)
