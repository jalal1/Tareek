"""Boundary freight demand.

Generates truck trips with at least one end outside the region (E->I, I->E,
E->E), so freeway and highway volumes are not short of the traffic real
corridors carry. Internal (I->I) freight is out of scope.

See docs/freight/design.md.
"""

from models.freight.cordons import (
    Cordon,
    CordonDetectionError,
    CordonDetector,
    DirectionalEnvelope,
    detector_from_config,
    weight_by_from_config,
    INBOUND,
    OUTBOUND,
    BIDIRECTIONAL,
    WEIGHT_BY_CAPACITY,
    WEIGHT_BY_HPMS_TRUCK_AADT,
)

from models.freight.hpms_match import (
    HPMSGeometryClient,
    HPMSSegment,
    build_truck_aadt_by_link,
    corridor_links,
    match_corridor_links,
    resolve_corridor_truck_aadt,
    cordon_bbox,
    directional_aadt,
    match_link_to_segment,
    parse_segments,
    resolve_truck_aadt_by_link,
)

from models.freight.truck_share import (
    HPMSClient,
    TruckShare,
    national_truck_share,
    resolve_truck_share,
    SOURCE_CONFIG_PINNED,
    SOURCE_HPMS_CACHE,
    SOURCE_HPMS_LIVE,
    SOURCE_NATIONAL_TABLE,
)

from models.freight.demand import (
    ClassShares,
    FreightDemand,
    assign_cordon_weights,
    crossing_factor,
    resolve_demand,
    CLASS_EXTERNAL_TO_INTERNAL,
    CLASS_INTERNAL_TO_EXTERNAL,
    CLASS_THROUGH,
    TRIP_CLASSES,
)
from models.freight.departure import (
    DepartureSampler,
    DepartureProfileError,
    normalise_profile,
    seconds_to_hms,
)
from models.freight.generator import (
    FreightGenerationError,
    FreightTrip,
    FreightTripGenerator,
    Zone,
    anchor_cordons_to_zones,
)
from models.freight.plans import generate_freight_plans, trips_to_plans
from models.freight.events import (
    EventsExtractionError,
    LinkVolumes,
    compare_against_observed,
    extract_freight_and_total,
    extract_link_volumes,
    freight_vehicle_ids,
    geh,
)
from models.freight.validation import (
    CheckResult,
    ValidationReport,
    summarise,
    validate_tier1,
    validate_tier2,
)
from models.freight.vehicles import build_vehicle_types, write_vehicles_file
from models.freight.classification import (
    ClassificationCoverage,
    check_coverage,
    compare_hourly_profile,
    validate_tier3,
)
from models.freight.reporting import (
    network_effect_digest,
    build_report,
    cordon_screenline_check,
    hourly_class_shares,
    percent_rmse,
    rmse_by_volume_group,
    trip_length_distribution,
    truck_percentage_by_class,
    vmt_by_class,
)

__all__ = [
    # cordons
    'Cordon',
    'CordonDetectionError',
    'CordonDetector',
    'DirectionalEnvelope',
    'detector_from_config',
    'weight_by_from_config',
    'INBOUND',
    'OUTBOUND',
    'BIDIRECTIONAL',
    'WEIGHT_BY_CAPACITY',
    'WEIGHT_BY_HPMS_TRUCK_AADT',
    # HPMS link matching (stage 9)
    'HPMSGeometryClient',
    'HPMSSegment',
    'build_truck_aadt_by_link',
    'corridor_links',
    'match_corridor_links',
    'resolve_corridor_truck_aadt',
    'cordon_bbox',
    'directional_aadt',
    'match_link_to_segment',
    'parse_segments',
    'resolve_truck_aadt_by_link',
    # truck share
    'HPMSClient',
    'TruckShare',
    'national_truck_share',
    'resolve_truck_share',
    'SOURCE_CONFIG_PINNED',
    'SOURCE_HPMS_CACHE',
    'SOURCE_HPMS_LIVE',
    'SOURCE_NATIONAL_TABLE',
    # demand
    'ClassShares',
    'FreightDemand',
    'assign_cordon_weights',
    'crossing_factor',
    'resolve_demand',
    'CLASS_EXTERNAL_TO_INTERNAL',
    'CLASS_INTERNAL_TO_EXTERNAL',
    'CLASS_THROUGH',
    'TRIP_CLASSES',
    # departure
    'DepartureSampler',
    'DepartureProfileError',
    'normalise_profile',
    'seconds_to_hms',
    # generation
    'FreightGenerationError',
    'FreightTrip',
    'FreightTripGenerator',
    'Zone',
    'anchor_cordons_to_zones',
    'generate_freight_plans',
    'trips_to_plans',
    # measurement
    'EventsExtractionError',
    'LinkVolumes',
    'compare_against_observed',
    'extract_freight_and_total',
    'extract_link_volumes',
    'freight_vehicle_ids',
    'geh',
    # validation
    'CheckResult',
    'ValidationReport',
    'summarise',
    'validate_tier1',
    'validate_tier2',
    # pce
    'build_vehicle_types',
    'write_vehicles_file',
    # tier 3
    'ClassificationCoverage',
    'check_coverage',
    'compare_hourly_profile',
    'validate_tier3',
    # reporting
    'build_report',
    'cordon_screenline_check',
    'network_effect_digest',
    'hourly_class_shares',
    'percent_rmse',
    'rmse_by_volume_group',
    'trip_length_distribution',
    'truck_percentage_by_class',
    'vmt_by_class',
]
