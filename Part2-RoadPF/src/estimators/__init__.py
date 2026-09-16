
from .dead_reckoning import DeadReckoningEstimator, DRConfig
from .kalman import KalmanEstimator, KFConfig
from .road_pf import RoadParticleFilter, PFConfig



ESTIMATORS = {
    "dead_reckoning": (DeadReckoningEstimator, DRConfig),
    "kalman": (KalmanEstimator, KFConfig),
    "road_pf": (RoadParticleFilter, PFConfig),
}


def build_estimator(name, graph, **kwargs):
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; have {sorted(ESTIMATORS)}")
    cls, cfg_cls = ESTIMATORS[name]
    valid = set(cfg_cls().__dict__.keys())
    cfg = cfg_cls(**{k: v for k, v in kwargs.items() if k in valid})
    return cls(graph, cfg)


__all__ = ["DeadReckoningEstimator", "DRConfig", "KalmanEstimator", "KFConfig",
           "RoadParticleFilter", "PFConfig", "ESTIMATORS", "build_estimator"]
