"""POI Weighting utility for non-work trip generation.

This module provides functionality to calculate importance weights for POIs
based on their tags and attributes. Weights are configurable via config.json
to allow tuning of POI attractiveness for different trip purposes.
"""
import json
from typing import Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class POIWeighting:
    """
    Calculate importance weights for POIs based on tags and attributes.

    Weights are used when sampling destination POIs for non-work trips,
    allowing more "important" POIs (e.g., large stores, branded locations)
    to be selected more frequently.

    Configuration is loaded from config['nonwork_purposes'][purpose]['poi_weighting'].
    """

    def __init__(self, config: Dict[str, Any], purpose: str):
        """
        Initialize POI weighting for a specific trip purpose.

        Args:
            config: Configuration dictionary
            purpose: Trip purpose ('Shopping', 'Recreation', etc.)
        """
        self.purpose = purpose
        self.config = config

        # Load weighting configuration for this purpose
        purpose_config = config.get('nonwork_purposes', {}).get(purpose, {})
        self.weighting_config = purpose_config.get('poi_weighting', {})
        self.enabled = self.weighting_config.get('enabled', False)

        if self.enabled:
            logger.debug(f"POI weighting enabled for {purpose}")
            logger.debug(f"  has_name multiplier: {self.weighting_config.get('has_name', 1.0)}")
            logger.debug(f"  brand_names: {list(self.weighting_config.get('brand_names', {}).keys())}")
            logger.debug(f"  shop_type weights: {self.weighting_config.get('shop_type', {})}")
        else:
            logger.debug(f"POI weighting disabled for {purpose} (uniform sampling)")

    def is_enabled(self) -> bool:
        """Check if POI weighting is enabled for this purpose."""
        return self.enabled

    def calculate_weight(self, poi: Dict[str, Any]) -> float:
        """
        Calculate weight for a POI based on tags and attributes.

        Args:
            poi: POI dictionary with keys: 'osm_id', 'name', 'activity', 'lat', 'lon', 'tags'
                 'tags' is a JSON string containing OSM tags

        Returns:
            Weight as a float (>= 1.0). Higher values mean more attractive.
            Returns 1.0 if weighting is disabled or no matching rules.
        """
        if not self.enabled:
            return 1.0

        weight = 1.0

        # Parse tags from JSON string
        tags = {}
        if poi.get('tags'):
            try:
                tags = json.loads(poi['tags'])
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"Could not parse tags for POI {poi.get('osm_id')}: {poi.get('tags')}")
                tags = {}

        # Has name bonus
        if poi.get('name'):
            has_name_multiplier = self.weighting_config.get('has_name', 1.0)
            weight *= has_name_multiplier

        # Brand name bonus (case-insensitive matching)
        brand_names = self.weighting_config.get('brand_names', {})
        if poi.get('name') and brand_names:
            name_lower = poi['name'].lower()
            for brand, brand_weight in brand_names.items():
                if brand.lower() == 'default':
                    continue
                if brand.lower() in name_lower:
                    weight *= brand_weight
                    break  # Only apply one brand bonus

        # Tag-based bonuses - dynamically apply weights for any *_type config keys
        # e.g., shop_type -> looks for OSM tag "shop", leisure_type -> "leisure", amenity_type -> "amenity"
        weight *= self._apply_tag_weights(tags)

        return weight

    def _apply_tag_weights(self, tags: Dict[str, str]) -> float:
        """
        Apply tag weights dynamically based on config keys ending with '_type'.

        For each key like 'shop_type', 'leisure_type', 'amenity_type' in config:
        - Extracts the OSM tag name (e.g., 'shop_type' -> 'shop')
        - Checks if POI has that OSM tag
        - Applies the corresponding weight from config

        Args:
            tags: Dict of OSM tags from the POI

        Returns:
            Combined multiplier from all matching tag weights
        """
        multiplier = 1.0

        # Find all config keys ending with '_type' (e.g., shop_type, leisure_type, amenity_type)
        for config_key, weight_map in self.weighting_config.items():
            if not config_key.endswith('_type') or not isinstance(weight_map, dict):
                continue

            # Extract OSM tag name: 'shop_type' -> 'shop', 'leisure_type' -> 'leisure'
            osm_tag = config_key[:-5]  # Remove '_type' suffix

            # Check if POI has this OSM tag
            if osm_tag in tags and weight_map:
                tag_value = tags[osm_tag]
                tag_weight = weight_map.get(tag_value, weight_map.get('default', 1.0))
                multiplier *= tag_weight

        return multiplier
