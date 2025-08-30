from .nerf import *
from .satnerf import *
from .snerf import *
from .satnerf_hash import *

def load_model(args):
    if args.model == "nerf":
        model = NeRF(layers=args.fc_layers, feat=args.fc_units)
    elif args.model == "s-nerf":
        model = ShadowNeRF(layers=args.fc_layers, feat=args.fc_units)
    elif args.model == "sat-nerf":
        model = SatNeRF(layers=args.fc_layers, feat=args.fc_units, t_embedding_dims=args.t_embbeding_tau)
    elif args.model == "sat-nerf-hash":
        model = SatNeRF_hash(layers=args.fc_layers, 
                             feat=args.fc_units, 
                             t_embedding_dims=args.t_embbeding_tau,
                             bounding_box=args.bounding_box,
                             n_levels=args.n_levels,
                             n_features_per_level=args.n_features_per_level,
                             log2_hashmap_size=args.log2_hashmap_size,
                             base_resolution=args.base_resolution,
                             finest_resolution=args.finest_resolution,
                             sh_degree=args.sh_degree)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return model
