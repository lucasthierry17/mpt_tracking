import dummy
import nikolucas_filters

# TODO: Add your filters here
filters = {
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "Nicolucas": {
        "color": [0.6, 0.6, 0.2],
        "constantposition": nikolucas_filters.KalmanFilter(),
        "constantvelocity": nikolucas_filters.ConstantVelocityKalmanFilter(),
        "constantvelocity2": nikolucas_filters.ConstantVelocity2(),
        "constantturn": nikolucas_filters.ConstantTurnRateFilter(4),
        "randomnoise": nikolucas_filters.RandomNoise(2, 2),
        "angular": nikolucas_filters.ExtendedKalmanFilter(),
    },
}
