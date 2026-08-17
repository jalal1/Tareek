"""Smoke tests for freight demand arithmetic, departure profiles and trips.

The arithmetic between an observed AADT and a simulated agent count passes
through four multiplications, and each of the design's three named traps is a
silent factor error. These tests pin all three.

See docs/freight/design.md §3 and §4.
"""

import numpy as np
import pytest

from models.freight.cordons import (
    BIDIRECTIONAL,
    Cordon,
    INBOUND,
    OUTBOUND,
    WEIGHT_BY_CAPACITY,
    WEIGHT_BY_HPMS_TRUCK_AADT,
    weight_by_from_config,
)
from models.freight.demand import (
    CAPACITY_TO_DAILY_VOLUME,
    CLASS_EXTERNAL_TO_INTERNAL,
    CLASS_INTERNAL_TO_EXTERNAL,
    CLASS_THROUGH,
    ClassShares,
    assign_cordon_weights,
    crossing_factor,
    probabilistic_round,
    resolve_demand,
)
from models.freight.departure import (
    BUSINESS_DAY_PROFILE,
    DepartureProfileError,
    DepartureSampler,
    PROFILE_BUSINESS_DAY,
    PROFILE_THROUGH,
    THROUGH_PROFILE,
    normalise_profile,
    seconds_to_hms,
)
from models.freight.generator import (
    FreightGenerationError,
    FreightTripGenerator,
    Zone,
    anchor_cordons_to_zones,
)
from models.freight.truck_share import national_truck_share


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cordons():
    """Four gateways, one per side, two each way."""
    return [
        Cordon('in_n', 0.0, 10_000.0, INBOUND, 'N', ['1'], capacity=8000, weight=5000),
        Cordon('in_s', 0.0, -10_000.0, INBOUND, 'S', ['2'], capacity=4000, weight=2500),
        Cordon('out_e', 10_000.0, 0.0, OUTBOUND, 'E', ['3'], capacity=8000, weight=5000),
        Cordon('out_w', -10_000.0, 0.0, OUTBOUND, 'W', ['4'], capacity=4000, weight=2500),
    ]


def _zones(n=40, seed=3):
    rng = np.random.default_rng(seed)
    zones = []
    for i in range(n):
        x = float(rng.uniform(-5000, 5000))
        y = float(rng.uniform(-5000, 5000))
        zones.append(Zone(f'z{i}', x, y, 33.5 + y / 1e6, -86.8 + x / 1e6,
                          attractor=float(rng.integers(10, 1000))))
    return zones


def _config(**overrides):
    config = {
        'plan_generation': {'scaling_factor': 1.0, 'random_seed': 42},
        'freight': {'enabled': True, 'demand_source': 'hpms_cordon',
                    'demand_scale': 1.0},
    }
    config['freight'].update(overrides)
    return config


# ---------------------------------------------------------------------------
# the crossing factor — trap 1
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_crossing_factor_at_the_default_through_share():
    """0.77, not 0.5. Using 0.5 would throw away a third of the demand."""
    assert crossing_factor(0.30) == pytest.approx(0.7692, abs=1e-4)


@pytest.mark.smoke
def test_crossing_factor_is_half_only_when_everything_is_through():
    assert crossing_factor(1.0) == pytest.approx(0.5)


@pytest.mark.smoke
def test_crossing_factor_is_one_when_nothing_is_through():
    """With no through trips every crossing is a trip: no correction at all."""
    assert crossing_factor(0.0) == pytest.approx(1.0)


@pytest.mark.smoke
@pytest.mark.parametrize('bad', [-0.1, 1.5])
def test_crossing_factor_rejects_impossible_shares(bad):
    with pytest.raises(ValueError):
        crossing_factor(bad)


# ---------------------------------------------------------------------------
# scaling — trap 2 — and probabilistic rounding — trap 3
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_scaling_factor_is_applied():
    """Freight generated at full volume would overstate trucks ~10x."""
    cordons = _cordons()
    full = resolve_demand(_config(), cordons, rng=np.random.default_rng(1))

    scaled_config = _config()
    scaled_config['plan_generation']['scaling_factor'] = 0.1
    scaled = resolve_demand(scaled_config, cordons, rng=np.random.default_rng(1))

    assert scaled.total_trips == pytest.approx(full.total_trips * 0.1, rel=0.15)


@pytest.mark.smoke
def test_demand_scale_multiplies_the_total():
    cordons = _cordons()
    base = resolve_demand(_config(), cordons, rng=np.random.default_rng(1))
    doubled = resolve_demand(_config(demand_scale=2.0), cordons,
                             rng=np.random.default_rng(1))
    assert doubled.total_trips == pytest.approx(base.total_trips * 2, rel=0.1)


@pytest.mark.smoke
def test_probabilistic_rounding_preserves_the_expected_total():
    """Flooring would bias every cordon's share down."""
    rng = np.random.default_rng(0)
    values = np.full(20_000, 2.5)
    rounded = probabilistic_round(values, rng)
    assert rounded.mean() == pytest.approx(2.5, abs=0.02)


@pytest.mark.smoke
def test_probabilistic_rounding_is_exact_on_integers():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 7.0, 0.0])
    assert list(probabilistic_round(values, rng)) == [1, 7, 0]


# ---------------------------------------------------------------------------
# class shares
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_class_shares_normalise_when_they_do_not_sum_to_one():
    """A config summing to 0.99 must not silently lose 1% of the demand."""
    shares = ClassShares(0.33, 0.33, 0.33)
    normalised = shares.normalised()
    assert sum(normalised.values()) == pytest.approx(1.0)


@pytest.mark.smoke
def test_class_shares_reject_negative_and_all_zero():
    with pytest.raises(ValueError):
        ClassShares(-0.1, 0.5, 0.5)
    with pytest.raises(ValueError):
        ClassShares(0.0, 0.0, 0.0)


@pytest.mark.smoke
def test_realised_class_split_matches_the_configuration():
    demand = resolve_demand(_config(), _cordons(), rng=np.random.default_rng(5))
    total = demand.total_trips
    for name, expected in ((CLASS_EXTERNAL_TO_INTERNAL, 0.35),
                           (CLASS_INTERNAL_TO_EXTERNAL, 0.35),
                           (CLASS_THROUGH, 0.30)):
        assert demand.trips_by_class[name] / total == pytest.approx(expected, abs=0.02)


# ---------------------------------------------------------------------------
# demand sources
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_absolute_source_uses_the_configured_total():
    config = _config(demand_source='absolute', trips_per_day=1000)
    demand = resolve_demand(config, _cordons(), rng=np.random.default_rng(1))
    assert demand.total_trips == pytest.approx(1000, rel=0.05)


@pytest.mark.smoke
def test_absolute_source_without_a_total_is_an_error():
    config = _config(demand_source='absolute')
    with pytest.raises(ValueError, match='trips_per_day'):
        resolve_demand(config, _cordons())


@pytest.mark.smoke
def test_car_share_source_scales_off_car_demand():
    share = national_truck_share(1, False)
    config = _config(demand_source='car_share')
    demand = resolve_demand(config, _cordons(), truck_share=share,
                            car_trips=100_000, rng=np.random.default_rng(1))
    assert demand.total_trips == pytest.approx(100_000 * share.total, rel=0.05)


@pytest.mark.smoke
def test_car_share_source_needs_car_trips():
    with pytest.raises(ValueError):
        resolve_demand(_config(demand_source='car_share'), _cordons())


@pytest.mark.smoke
def test_unweighted_cordons_are_an_error_not_zero_freight():
    """Silently generating nothing is the failure the design forbids."""
    cordons = [Cordon('a', 0, 0, INBOUND, 'N', ['1'], capacity=1000, weight=0.0)]
    with pytest.raises(ValueError, match='weight'):
        resolve_demand(_config(), cordons)


@pytest.mark.smoke
def test_invalid_demand_source_rejected():
    with pytest.raises(ValueError):
        resolve_demand(_config(demand_source='nonsense'), _cordons())


# ---------------------------------------------------------------------------
# cordon weighting — the units trap
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_observed_volume_beats_capacity():
    cordons = _cordons()
    stats = assign_cordon_weights(cordons, truck_aadt_by_link={'1': 4321.0})
    assert stats['n_observed'] == 1
    assert cordons[0].weight == pytest.approx(4321.0)


@pytest.mark.smoke
def test_capacity_fallback_is_converted_to_a_daily_volume():
    """Capacity is veh/hour, AADT is a daily count.

    Mixing them without conversion would make capacity-weighted cordons carry
    roughly an eighth of their proper share against observed ones.
    """
    share = national_truck_share(1, False)
    cordons = [Cordon('a', 0, 0, INBOUND, 'N', ['1'], capacity=1000.0)]
    assign_cordon_weights(cordons, truck_share=share)

    expected = 1000.0 * CAPACITY_TO_DAILY_VOLUME * share.total
    assert cordons[0].weight == pytest.approx(expected)
    assert cordons[0].weight > 1000.0 * share.total


@pytest.mark.smoke
def test_weighting_reports_how_many_cordons_used_each_route():
    cordons = _cordons()
    stats = assign_cordon_weights(cordons, truck_aadt_by_link={'1': 100.0, '3': 200.0})
    assert (stats['n_observed'], stats['n_fallback']) == (2, 2)


@pytest.mark.smoke
def test_weight_by_capacity_ignores_observed_volumes():
    """The ablation control: 'capacity' must not quietly use observations.

    Its whole purpose is to answer "how much of the spatial distribution is the
    data doing?", which it cannot do if an observed volume still leaks in.
    """
    cordons = _cordons()
    stats = assign_cordon_weights(
        cordons,
        truck_aadt_by_link={'1': 100.0, '3': 200.0},
        weight_by=WEIGHT_BY_CAPACITY,
    )
    assert (stats['n_observed'], stats['n_fallback']) == (0, 4)
    assert stats['weight_by'] == WEIGHT_BY_CAPACITY


@pytest.mark.smoke
def test_weight_by_records_which_route_was_taken():
    """The summary must say how cordons were weighted, not leave it inferred."""
    stats = assign_cordon_weights(_cordons(), truck_aadt_by_link={'1': 100.0})
    assert stats['weight_by'] == WEIGHT_BY_HPMS_TRUCK_AADT


@pytest.mark.smoke
def test_unrecognised_weight_by_is_rejected():
    """A misspelling would change every truck's origin while reporting success."""
    with pytest.raises(ValueError, match='weight_by'):
        assign_cordon_weights(_cordons(), weight_by='hpms')


@pytest.mark.smoke
def test_weight_by_from_config_defaults_and_validates():
    assert weight_by_from_config({}) == WEIGHT_BY_HPMS_TRUCK_AADT
    assert weight_by_from_config(
        {'freight': {'cordon': {'weight_by': 'capacity'}}}) == WEIGHT_BY_CAPACITY
    with pytest.raises(ValueError, match='weight_by'):
        weight_by_from_config({'freight': {'cordon': {'weight_by': 'nope'}}})


# ---------------------------------------------------------------------------
# departure profiles
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_profiles_are_normalised_not_assumed_to_sum_to_one():
    """The shipped defaults sum to 0.935 and 0.985 — read off a chart."""
    assert sum(BUSINESS_DAY_PROFILE) != pytest.approx(1.0)
    assert normalise_profile(BUSINESS_DAY_PROFILE).sum() == pytest.approx(1.0)
    assert normalise_profile(THROUGH_PROFILE).sum() == pytest.approx(1.0)


@pytest.mark.smoke
@pytest.mark.parametrize('bad,reason', [
    ([0.1] * 23, 'wrong length'),
    ([0.1] * 25, 'wrong length'),
    ([-0.1] + [0.1] * 23, 'negative'),
    ([0.0] * 24, 'sums to zero'),
])
def test_invalid_profiles_are_rejected(bad, reason):
    with pytest.raises(DepartureProfileError):
        normalise_profile(bad)


@pytest.mark.smoke
def test_through_trips_use_the_flat_profile():
    sampler = DepartureSampler()
    assert sampler.profile_for_class(CLASS_THROUGH) == PROFILE_THROUGH
    assert sampler.profile_for_class(CLASS_EXTERNAL_TO_INTERNAL) == PROFILE_BUSINESS_DAY
    assert sampler.profile_for_class(CLASS_INTERNAL_TO_EXTERNAL) == PROFILE_BUSINESS_DAY


@pytest.mark.smoke
def test_business_day_is_peakier_than_through():
    """The whole reason for two profiles rather than one global curve."""
    sampler = DepartureSampler()
    business = np.array(sampler.profiles[PROFILE_BUSINESS_DAY])
    through = np.array(sampler.profiles[PROFILE_THROUGH])
    assert business.max() - business.min() > 2 * (through.max() - through.min())


@pytest.mark.smoke
def test_sampled_departures_follow_the_profile():
    sampler = DepartureSampler()
    rng = np.random.default_rng(11)
    seconds = sampler.sample(CLASS_EXTERNAL_TO_INTERNAL, rng, 20_000)
    realised = np.array(sampler.realised_distribution(seconds))
    configured = np.array(sampler.profiles[PROFILE_BUSINESS_DAY])
    assert np.abs(realised - configured).max() < 0.01


@pytest.mark.smoke
def test_departures_are_jittered_within_the_hour():
    """Without jitter MATSim releases a whole hour's trucks in one second."""
    sampler = DepartureSampler()
    seconds = sampler.sample(CLASS_THROUGH, np.random.default_rng(2), 500)
    within_hour = seconds % 3600
    assert len(np.unique(within_hour)) > 400


@pytest.mark.smoke
def test_seconds_to_hms_does_not_wrap_past_midnight():
    assert seconds_to_hms(0) == '00:00:00'
    assert seconds_to_hms(3661) == '01:01:01'
    assert seconds_to_hms(25 * 3600) == '25:00:00'


@pytest.mark.smoke
def test_custom_profile_from_config_is_used():
    profile = [0.0] * 24
    profile[7] = 1.0
    config = {'freight': {'departure_distribution': {'business_day': profile}}}
    sampler = DepartureSampler.from_config(config)
    seconds = sampler.sample(CLASS_EXTERNAL_TO_INTERNAL, np.random.default_rng(1), 50)
    assert ((seconds >= 7 * 3600) & (seconds < 8 * 3600)).all()


# ---------------------------------------------------------------------------
# anchoring and trip generation
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_anchoring_records_distance_to_the_nearest_zone():
    cordons = _cordons()
    stats = anchor_cordons_to_zones(cordons, _zones())
    for cordon in cordons:
        assert cordon.zone_id is not None
        assert cordon.anchor_distance_m >= 0
    assert stats['anchor_distance_m']['max'] >= stats['anchor_distance_m']['min']


@pytest.mark.smoke
def test_anchoring_without_zones_raises():
    with pytest.raises(FreightGenerationError):
        anchor_cordons_to_zones(_cordons(), [])


@pytest.mark.smoke
def test_generator_requires_cordons_and_zones():
    with pytest.raises(FreightGenerationError):
        FreightTripGenerator([], _zones(), _config())
    with pytest.raises(FreightGenerationError):
        FreightTripGenerator(_cordons(), [], _config())


@pytest.mark.smoke
def test_trip_ends_respect_cordon_direction():
    """E->I must start on an inbound cordon, I->E end on an outbound one.

    Otherwise trucks are injected the wrong way up a divided highway.
    """
    cordons = _cordons()
    zones = _zones()
    anchor_cordons_to_zones(cordons, zones)
    config = _config()
    demand = resolve_demand(config, cordons, rng=np.random.default_rng(4))
    generator = FreightTripGenerator(cordons, zones, config,
                                     rng=np.random.default_rng(4))
    trips = generator.generate(demand)

    inbound = {c.cordon_id for c in cordons if c.accepts_entry}
    outbound = {c.cordon_id for c in cordons if c.accepts_exit}

    for trip in trips:
        if trip.trip_class == CLASS_EXTERNAL_TO_INTERNAL:
            assert trip.origin_cordon in inbound
            assert trip.dest_zone is not None
        elif trip.trip_class == CLASS_INTERNAL_TO_EXTERNAL:
            assert trip.dest_cordon in outbound
            assert trip.origin_zone is not None


@pytest.mark.smoke
def test_through_trips_use_two_different_cordons():
    cordons = _cordons()
    zones = _zones()
    anchor_cordons_to_zones(cordons, zones)
    config = _config()
    demand = resolve_demand(config, cordons, rng=np.random.default_rng(6))
    generator = FreightTripGenerator(cordons, zones, config,
                                     rng=np.random.default_rng(6))
    through = [t for t in generator.generate(demand)
               if t.trip_class == CLASS_THROUGH]

    assert through
    for trip in through:
        assert trip.origin_cordon != trip.dest_cordon


@pytest.mark.smoke
def test_every_requested_trip_is_generated():
    """The generator must not quietly produce fewer trips than demanded.

    An over-strict interior-crossing test once dropped every E->E trip while
    the run still reported success — 30% of the demand gone, silently. The
    count is the assertion; the geometry is a means, not the contract.
    """
    cordons = _cordons()
    zones = _zones()
    anchor_cordons_to_zones(cordons, zones)
    config = _config()
    demand = resolve_demand(config, cordons, rng=np.random.default_rng(6))
    generator = FreightTripGenerator(cordons, zones, config,
                                     rng=np.random.default_rng(6))
    trips = generator.generate(demand)

    assert len(trips) == demand.total_trips
    counts = {name: 0 for name in (CLASS_EXTERNAL_TO_INTERNAL,
                                   CLASS_INTERNAL_TO_EXTERNAL, CLASS_THROUGH)}
    for trip in trips:
        counts[trip.trip_class] += 1
    assert counts == demand.trips_by_class


@pytest.mark.smoke
def test_perpendicular_gateways_count_as_crossing():
    """A truck entering north and leaving east crosses the region.

    The metric version of this test rejected exactly this case: the chord
    between two perpendicular peripheral cordons passes at r/sqrt(2), further
    from the centre than the zones spread.
    """
    cordons = [
        Cordon('in_n', 0.0, 10_000.0, INBOUND, 'N', ['1'], capacity=8000, weight=5000),
        Cordon('out_e', 10_000.0, 0.0, OUTBOUND, 'E', ['2'], capacity=8000, weight=5000),
    ]
    zones = _zones()
    generator = FreightTripGenerator(cordons, zones, _config(),
                                     rng=np.random.default_rng(1))
    pairs = generator._build_through_pairs()

    assert pairs
    assert any(e.cordon_id == 'in_n' and x.cordon_id == 'out_e'
               for e, x, _ in pairs)


@pytest.mark.smoke
def test_same_side_gateways_do_not_pair():
    """Two neighbours on one edge would be a hop along the boundary."""
    cordons = [
        Cordon('in_n1', -1000.0, 10_000.0, INBOUND, 'N', ['1'],
               capacity=8000, weight=5000),
        Cordon('out_n2', 1000.0, 10_000.0, OUTBOUND, 'N', ['2'],
               capacity=8000, weight=5000),
        # a genuine opposite gateway, so the fallback does not engage
        Cordon('out_s', 0.0, -10_000.0, OUTBOUND, 'S', ['3'],
               capacity=8000, weight=5000),
    ]
    generator = FreightTripGenerator(cordons, _zones(), _config(),
                                     rng=np.random.default_rng(1))
    pairs = generator._build_through_pairs()

    paired = {(e.cordon_id, x.cordon_id) for e, x, _ in pairs}
    assert ('in_n1', 'out_n2') not in paired
    assert ('in_n1', 'out_s') in paired


@pytest.mark.smoke
def test_generation_is_reproducible_from_the_seed():
    cordons, zones = _cordons(), _zones()
    anchor_cordons_to_zones(cordons, zones)
    config = _config()

    def run():
        demand = resolve_demand(config, cordons, rng=np.random.default_rng(9))
        generator = FreightTripGenerator(cordons, zones, config,
                                         rng=np.random.default_rng(9))
        return [(t.trip_class, t.origin_cordon, round(t.departure_seconds, 6))
                for t in generator.generate(demand)]

    assert run() == run()


@pytest.mark.smoke
def test_zone_sampling_prefers_nearby_and_attractive_zones():
    """The singly-constrained step: attractor x friction(distance)."""
    cordon = Cordon('c', 0.0, 0.0, INBOUND, 'N', ['1'], capacity=1000, weight=1000)
    near = Zone('near', 1000.0, 0.0, 33.5, -86.8, attractor=100.0)
    far = Zone('far', 60_000.0, 0.0, 33.9, -86.2, attractor=100.0)
    generator = FreightTripGenerator([cordon], [near, far], _config(),
                                     rng=np.random.default_rng(0))
    weights = generator._zone_weights(cordon)
    assert weights[0] > weights[1]


@pytest.mark.smoke
def test_bidirectional_cordon_serves_both_directions():
    cordon = Cordon('both', 0.0, 10_000.0, BIDIRECTIONAL, 'N', ['1'],
                    capacity=4000, weight=1000)
    assert cordon.accepts_entry and cordon.accepts_exit
