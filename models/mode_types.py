"""
Mode types and configuration for multi-modal transportation.

This module is the single source of truth for canonical transportation modes.
All surveys and plan generators should import mode constants from here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any


# ── Canonical transport modes (single source of truth) ──────────────────────
# All surveys must map their raw mode labels to these values.
MODE_CAR = 'Car'
MODE_BUS = 'Bus'
MODE_RAIL = 'Rail'
MODE_WALK = 'Walk'
MODE_BIKE = 'Bike'
MODE_SCHOOL_BUS = 'School Bus'
MODE_RIDESHARE = 'Rideshare'
MODE_OTHER = 'Other'

CANONICAL_MODES = {
    MODE_CAR, MODE_BUS, MODE_RAIL, MODE_WALK,
    MODE_BIKE, MODE_SCHOOL_BUS, MODE_RIDESHARE, MODE_OTHER,
}


class ModeType(Enum):
    """Canonical transportation modes as enum."""
    CAR = "car"
    BUS = "bus"
    RAIL = "rail"
    WALK = "walk"
    BIKE = "bike"
    SCHOOL_BUS = "school_bus"
    RIDESHARE = "rideshare"
    OTHER = "other"

    @classmethod
    def from_survey_mode(cls, survey_mode: str) -> 'ModeType':
        """
        Map survey canonical mode string (e.g., 'Car', 'Bus') to ModeType enum.

        Args:
            survey_mode: Mode string from survey data (uses MODE_* constants)

        Returns:
            Corresponding ModeType enum value
        """
        mapping = {
            MODE_CAR: cls.CAR,
            MODE_BUS: cls.BUS,
            MODE_RAIL: cls.RAIL,
            MODE_WALK: cls.WALK,
            MODE_BIKE: cls.BIKE,
            MODE_SCHOOL_BUS: cls.SCHOOL_BUS,
            MODE_RIDESHARE: cls.RIDESHARE,
            MODE_OTHER: cls.OTHER,
        }
        return mapping.get(survey_mode, cls.OTHER)

    def to_output_mode(self, output_format: str = 'matsim') -> str:
        """
        Convert to output format mode string.

        Args:
            output_format: Target output format ('matsim', 'sumo', etc.)

        Returns:
            Mode string appropriate for the output format
        """
        if output_format == 'matsim':
            # MATSim uses 'pt' for public transit modes
            matsim_mapping = {
                ModeType.CAR: 'car',
                ModeType.BUS: 'pt',
                ModeType.RAIL: 'pt',
                ModeType.WALK: 'walk',
                ModeType.BIKE: 'bike',
                ModeType.SCHOOL_BUS: 'pt',
                ModeType.RIDESHARE: 'car',
                ModeType.OTHER: 'car',
            }
            return matsim_mapping.get(self, 'car')
        # Default: return enum value
        return self.value

    def is_transit(self) -> bool:
        """Check if this is a public transit mode."""
        return self in {ModeType.BUS, ModeType.RAIL, ModeType.SCHOOL_BUS}


@dataclass
class ModeConfig:
    """Configuration for a single mode, loaded from config.json.

    Rate configuration fields (for Phase 2 mode choice):
    - survey_rate: "auto" to compute from survey, or float value (0.0-1.0)
    - config_rate: Target rate override, or None to skip blending
    - blend_weight: How to blend survey vs config (0.0=survey only, 1.0=config only)
    - share_adjustment: Additive adjustment to final rate (e.g., +0.05 = +5 points)
    """
    mode_type: ModeType
    matsim_mode: str              # Output mode string for MATSim (e.g., "pt")
    availability_type: str        # "universal", "gtfs", "distance", "zone_list"
    availability_params: Dict[str, Any]  # Type-specific parameters
    enabled: bool                 # Whether this mode is active (default True)

    # Rate configuration (Phase 2)
    survey_rate: str | float      # "auto" or fixed float value
    config_rate: Optional[float]  # Target rate for blending, None = no override
    blend_weight: float           # 0.0 = pure survey, 1.0 = pure config
    share_adjustment: float       # Additive adjustment after blending

    @classmethod
    def from_config(cls, mode_name: str, config: Dict[str, Any]) -> 'ModeConfig':
        """
        Create ModeConfig from config.json entry.

        Args:
            mode_name: Mode name key (e.g., "car", "bus")
            config: Mode configuration dict from config.json

        Returns:
            ModeConfig instance
        """
        # Parse mode type from name
        try:
            mode_type = ModeType(mode_name.lower())
        except ValueError:
            mode_type = ModeType.OTHER

        # Parse availability
        availability = config.get('availability', 'universal')
        if isinstance(availability, str):
            availability_type = availability
            availability_params = {}
        else:
            availability_type = availability.get('type', 'universal')
            availability_params = {k: v for k, v in availability.items() if k != 'type'}

        # Parse rate configuration (with defaults for backwards compatibility)
        survey_rate = config.get('survey_rate', 'auto')
        config_rate = config.get('config_rate', None)
        blend_weight = config.get('blend_weight', 0.0)
        share_adjustment = config.get('share_adjustment', 0.0)

        return cls(
            mode_type=mode_type,
            matsim_mode=config.get('matsim_mode', mode_type.to_output_mode('matsim')),
            availability_type=availability_type,
            availability_params=availability_params,
            enabled=config.get('enabled', True),
            survey_rate=survey_rate,
            config_rate=config_rate,
            blend_weight=blend_weight,
            share_adjustment=share_adjustment,
        )


def get_default_car_config() -> Dict[str, Any]:
    """Return default car mode configuration."""
    return {
        "car": {
            "matsim_mode": "car",
            "availability": "universal"
        }
    }
