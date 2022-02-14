from config import Arguments
import yaml

config = Arguments()

with open('./hyper_params.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)