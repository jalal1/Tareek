"""Cordon detection — where trucks enter and leave the simulated region.

A cordon is the freight equivalent of a traditional model's *external station*:
a point on the region boundary where an external trip crosses. Standard practice
places external stations by hand from a map. We derive them from the network, so
the module works for any region with no manual setup.

Two obvious methods were measured against a real cached network (Anoka County,
MN; 40,877 nodes, 92,422 links) and **both fail**. They are recorded here because
each looks correct until it is run:

*Severed stubs.* Measured: **zero** nodes lack an incoming or outgoing link.
(The test network was never cleaned, so the stubs are absent from the OSM
extract itself; where ``clean_network`` runs, its strongly-connected-component
filter removes any that did survive.) Either way this detector finds nothing,
silently.

*A band around the network extent.* A min/max rectangle is defined by its
outliers, and the outliers are noise. Measured: 1,371 nodes lie within 2 km of
the south edge but only **3** near the north and **13** near the west, so a 2 km
band puts every cordon on one side of the region. The motorway network spans
39x42 km inside a 45x51 km extent — the real road edge is 3-6 km inside the
rectangle on most sides.

What works is the conjunction of a topological test and a geometric one:

1. **Terminus.** A motorway-class link whose downstream node carries no onward
   motorway-class link is where the freeway leaves the modelled area. This needs
   no geometry at all.
2. **Peripheral.** Terminus alone is not enough — measured, **60%** of raw
   termini (233 of 391) sit below the peripherality threshold, where a capacity
   filter breaks a chain mid-region rather than at a gateway. A terminus counts
   only if it lies near the network's outer envelope *in its own direction*,
   because the network is not a disc and a single global radius admits interior
   links in the wide directions while cutting real gateways in the narrow ones.

Measured at the defaults: 36 inbound and 37 outbound cordons spread across all
eight compass directions, with the largest roads (capacity 10000, 9000, 7500)
retained at every threshold tested.

See docs/freight/design.md §2.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


# Which way a cordon carries traffic relative to the region.
#
# Direction is not cosmetic. A truck entering the region and one leaving it use
# opposite carriageways, which are separate links. Injecting an E->I trip onto
# an outbound link sends it the wrong way up a divided highway: it either fails
# to route or loads the wrong side of the corridor.
INBOUND = 'inbound'        # traffic enters the region here (E->I starts here)
OUTBOUND = 'outbound'      # traffic leaves the region here (I->E ends here)
BIDIRECTIONAL = 'both'     # an undivided road, usable either way


class CordonDetectionError(RuntimeError):
    """Raised when a region yields no usable cordons.

    Deliberately fatal. A region with no detected gateway is a bug or a badly
    set threshold, never a legitimate result — and generating zero freight while
    reporting success is the failure mode hardest to notice.
    """


@dataclass
class DirectionalEnvelope:
    """How far the network reaches in each direction from its centre.

    The network is not a disc: measured on the test network the envelope runs from
    13.5 km in the narrowest direction to 35.2 km in the widest. Comparing every
    node against one global radius would therefore admit interior links where
    the network is wide and reject real gateways where it is narrow. Binning by
    bearing removes that bias.

    A high percentile rather than the maximum, for the same reason the extent
    rectangle failed: the furthest node in a bin is often a stray artefact.
    """
    centroid: Tuple[float, float]
    radii: np.ndarray          # one radius per angular bin
    n_bins: int

    @classmethod
    def build(cls, points: np.ndarray, n_bins: int = 36,
              percentile: float = 99.0) -> 'DirectionalEnvelope':
        if len(points) == 0:
            raise CordonDetectionError("Cannot build an envelope with no points")

        centroid = points.mean(axis=0)
        offsets = points - centroid
        bearings = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360.0
        radii_all = np.hypot(offsets[:, 0], offsets[:, 1])
        bin_index = np.minimum((bearings // (360.0 / n_bins)).astype(int),
                               n_bins - 1)

        radii = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_index == b
            if mask.any():
                radii[b] = np.percentile(radii_all[mask], percentile)

        # An empty bin (a direction with no nodes at all) would divide by zero.
        # Fill it from the nearest non-empty bin so peripherality stays defined
        # everywhere rather than silently scoring 0.
        if (radii == 0).any() and (radii > 0).any():
            filled = radii > 0
            for b in np.flatnonzero(~filled):
                offset = np.abs(((np.flatnonzero(filled) - b + n_bins // 2)
                                 % n_bins) - n_bins // 2)
                radii[b] = radii[np.flatnonzero(filled)[offset.argmin()]]

        return cls(centroid=(float(centroid[0]), float(centroid[1])),
                   radii=radii, n_bins=n_bins)

    def peripherality(self, x: float, y: float) -> float:
        """How far out a point sits, as a fraction of the network's reach.

        1.0 means the point is on the envelope; 0.0 is the centre. Values above
        1.0 are possible for the outliers the percentile clipped, and are not
        capped — being further out than the envelope is not a problem.
        """
        dx = x - self.centroid[0]
        dy = y - self.centroid[1]
        bearing = math.degrees(math.atan2(dy, dx)) % 360.0
        index = min(int(bearing // (360.0 / self.n_bins)), self.n_bins - 1)
        reach = self.radii[index]
        if reach <= 0:
            return 0.0
        return math.hypot(dx, dy) / reach

    def compass(self, x: float, y: float) -> str:
        """Which compass octant a point lies in, for reporting."""
        dx = x - self.centroid[0]
        dy = y - self.centroid[1]
        bearing = math.degrees(math.atan2(dy, dx)) % 360.0
        octants = ('E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE')
        return octants[int(((bearing + 22.5) % 360.0) // 45.0)]


@dataclass
class Cordon:
    """One gateway on the region boundary.

    Attributes:
        cordon_id: stable identifier, derived from the representative link.
        x, y: the crossing point in network coordinates (metres).
        direction: INBOUND, OUTBOUND or BIDIRECTIONAL.
        compass: which side of the region it sits on, for reporting.
        link_ids: every network link merged into this cordon.
        capacity: summed ``lanes x capacity`` over the merged links. The
            fallback weight, used when no observed truck volume is available.
        weight: the sampling weight actually used. Set by the demand stage from
            observed truck volume where possible, otherwise from capacity.
        peripherality: how far out the crossing sits, 1.0 = on the envelope.
        zone_id: the block group this cordon is anchored to (stage 4).
        anchor_distance_m: how far that zone is from the crossing point. A large
            value means the corridor is being loaded from the wrong place, and
            is the first thing to check when a freeway is short of traffic.
    """
    cordon_id: str
    x: float
    y: float
    direction: str
    compass: str
    link_ids: List[str] = field(default_factory=list)
    capacity: float = 0.0
    weight: float = 0.0
    peripherality: float = 0.0
    zone_id: Optional[str] = None
    anchor_distance_m: Optional[float] = None

    @property
    def accepts_entry(self) -> bool:
        """Whether an E->I trip may originate here."""
        return self.direction in (INBOUND, BIDIRECTIONAL)

    @property
    def accepts_exit(self) -> bool:
        """Whether an I->E trip may terminate here."""
        return self.direction in (OUTBOUND, BIDIRECTIONAL)

    def to_dict(self) -> Dict:
        """Plain-dict form for freight_summary.json."""
        return {
            'cordon_id': self.cordon_id,
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'direction': self.direction,
            'compass': self.compass,
            'n_links': len(self.link_ids),
            'link_ids': self.link_ids,
            'capacity': round(self.capacity, 1),
            'weight': round(self.weight, 6),
            'peripherality': round(self.peripherality, 3),
            'zone_id': self.zone_id,
            'anchor_distance_m': (round(self.anchor_distance_m, 1)
                                  if self.anchor_distance_m is not None else None),
        }


@dataclass(frozen=True)
class _Terminus:
    """A motorway link that starts or ends the motorway chain, before dedupe."""
    link_id: str
    x: float                   # the crossing point (the terminal node)
    y: float
    direction: str
    capacity: float
    peripherality: float
    compass: str


class CordonDetector:
    """Finds boundary crossings in a MATSim network.

    Usage::

        detector = CordonDetector()
        cordons = detector.detect(network_path)

    The detector holds only parameters, never the network, so one instance can
    process several regions and the parsed network is free to be released.
    """

    def __init__(
        self,
        min_freespeed_ms: float = 22.0,
        min_capacity_vph: float = 1500.0,
        min_peripherality: float = 0.6,
        dedupe_radius_m: float = 1500.0,
        envelope_bins: int = 36,
        envelope_percentile: float = 99.0,
        mode: str = 'car',
        fail_if_none_found: bool = True,
    ):
        """
        Args:
            min_freespeed_ms: motorway-class filter (~80 km/h at the default).
            min_capacity_vph: its capacity counterpart. Together these define
                what counts as a through corridor rather than a local street.
            min_peripherality: how far out a terminus must sit, as a fraction of
                the network's reach in that direction. The measured sensitivity
                is smooth — 0.4/0.6/0.8 give 98/73/39 cordons on the test network,
                all retaining the interstates — so this is a soft knob, not a
                cliff. 0.6 keeps every compass direction populated.
            dedupe_radius_m: crossings closer than this, heading the same way,
                merge into one cordon.
            envelope_bins: angular resolution of the directional envelope. 36
                bins is 10 degrees each.
            envelope_percentile: which percentile of node radius defines the
                network's reach in a bin. Below 100 on purpose: the furthest
                node in a direction is often an artefact.
            mode: the network mode a cordon must carry.
            fail_if_none_found: raise rather than return an empty list.
        """
        if not 0.0 < min_peripherality <= 2.0:
            raise ValueError(
                f"min_peripherality must be in (0, 2], got {min_peripherality}")
        if dedupe_radius_m < 0:
            raise ValueError(f"dedupe_radius_m must be >= 0, got {dedupe_radius_m}")
        if envelope_bins < 4:
            raise ValueError(f"envelope_bins must be >= 4, got {envelope_bins}")

        self.min_freespeed_ms = float(min_freespeed_ms)
        self.min_capacity_vph = float(min_capacity_vph)
        self.min_peripherality = float(min_peripherality)
        self.dedupe_radius_m = float(dedupe_radius_m)
        self.envelope_bins = int(envelope_bins)
        self.envelope_percentile = float(envelope_percentile)
        self.mode = mode
        self.fail_if_none_found = fail_if_none_found

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, network_path: Path) -> List[Cordon]:
        """Find the cordons in a MATSim network file."""
        nodes, links = self._parse_network(Path(network_path))
        envelope = DirectionalEnvelope.build(
            np.array(list(nodes.values())),
            n_bins=self.envelope_bins,
            percentile=self.envelope_percentile,
        )
        logger.info(
            f"Network: {len(nodes):,} nodes, {len(links):,} links; reach "
            f"{envelope.radii.min() / 1000:.1f}-{envelope.radii.max() / 1000:.1f} km "
            f"by direction"
        )

        termini = self._find_termini(nodes, links, envelope)
        logger.info(
            f"Motorway termini past peripherality {self.min_peripherality}: "
            f"{sum(1 for t in termini if t.direction == INBOUND)} inbound, "
            f"{sum(1 for t in termini if t.direction == OUTBOUND)} outbound"
        )

        cordons = self._deduplicate(termini)
        logger.info(f"Cordons after deduplication: {len(cordons)}")

        if not cordons:
            message = (
                f"No freight cordons found in {network_path}. Every region on a "
                f"road network has gateways, so this is a configuration or data "
                f"problem, not a valid result. Check freight.cordon: "
                f"min_peripherality={self.min_peripherality}, "
                f"min_freespeed_ms={self.min_freespeed_ms}, "
                f"min_capacity_vph={self.min_capacity_vph} against the "
                f"network's actual link attributes."
            )
            if self.fail_if_none_found:
                raise CordonDetectionError(message)
            logger.warning(message)

        return cordons

    def describe(self, cordons: Sequence[Cordon]) -> Dict:
        """Summary of a cordon set, for freight_summary.json and for logs."""
        by_compass: Dict[str, int] = defaultdict(int)
        by_direction: Dict[str, int] = defaultdict(int)
        for cordon in cordons:
            by_compass[cordon.compass] += 1
            by_direction[cordon.direction] += 1
        return {
            'n_cordons': len(cordons),
            'n_entry': sum(1 for c in cordons if c.accepts_entry),
            'n_exit': sum(1 for c in cordons if c.accepts_exit),
            'by_compass': dict(sorted(by_compass.items())),
            'by_direction': dict(sorted(by_direction.items())),
            'total_capacity': round(sum(c.capacity for c in cordons), 1),
            'detector': {
                'min_freespeed_ms': self.min_freespeed_ms,
                'min_capacity_vph': self.min_capacity_vph,
                'min_peripherality': self.min_peripherality,
                'dedupe_radius_m': self.dedupe_radius_m,
            },
        }

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_network(
        network_path: Path,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, str]]]:
        """Read nodes and links from a MATSim network.xml.

        Uses iterparse and clears each element once read: the test network
        is 51 MB of XML, and a full DOM of it is a needless resident cost inside
        a run that is already holding the OD matrix.
        """
        if not network_path.exists():
            raise CordonDetectionError(f"Network file not found: {network_path}")

        nodes: Dict[str, Tuple[float, float]] = {}
        links: List[Dict[str, str]] = []

        for _, elem in ET.iterparse(str(network_path), events=('end',)):
            if elem.tag == 'node':
                try:
                    nodes[elem.get('id')] = (float(elem.get('x')),
                                             float(elem.get('y')))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping node with bad coordinates: {elem.attrib}")
                elem.clear()
            elif elem.tag == 'link':
                links.append(dict(elem.attrib))
                elem.clear()

        if not nodes:
            raise CordonDetectionError(f"Network has no nodes: {network_path}")
        if not links:
            raise CordonDetectionError(f"Network has no links: {network_path}")

        return nodes, links

    def _is_corridor(self, link: Dict[str, str]) -> Optional[Tuple[float, float]]:
        """Return (capacity, lanes) when a link is a through corridor, else None."""
        modes = [m.strip() for m in (link.get('modes') or '').split(',')]
        if self.mode not in modes:
            return None
        try:
            freespeed = float(link.get('freespeed', 0.0))
            capacity = float(link.get('capacity', 0.0))
            lanes = float(link.get('permlanes', 1.0))
        except (TypeError, ValueError):
            return None
        if freespeed < self.min_freespeed_ms or capacity < self.min_capacity_vph:
            return None
        return capacity, max(1.0, lanes)

    def _find_termini(
        self,
        nodes: Dict[str, Tuple[float, float]],
        links: Iterable[Dict[str, str]],
        envelope: DirectionalEnvelope,
    ) -> List[_Terminus]:
        """Find corridor links that begin or end the corridor network.

        A link whose *from* node has no incoming corridor link is where traffic
        enters the modelled area — an inbound gateway. A link whose *to* node has
        no outgoing corridor link is where it leaves — an outbound gateway. The
        peripherality test then discards the interior breaks: measured on Anoka,
        233 of 391 raw termini (60%) fall below the threshold, where a capacity
        filter interrupts a chain mid-region rather than at a boundary.
        """
        corridor: Dict[str, Tuple[str, str, float]] = {}
        # Node -> the set of corridor links entering / leaving it. Sets rather
        # than booleans because the continuity test below has to exclude one
        # specific link, not just ask whether any exists.
        incoming: Dict[str, set] = defaultdict(set)
        outgoing: Dict[str, set] = defaultdict(set)

        for link in links:
            attributes = self._is_corridor(link)
            if attributes is None:
                continue
            capacity, lanes = attributes
            from_node, to_node = link.get('from'), link.get('to')
            if from_node not in nodes or to_node not in nodes:
                continue
            link_id = link.get('id')
            corridor[link_id] = (from_node, to_node, capacity * lanes)
            outgoing[from_node].add(link_id)
            incoming[to_node].add(link_id)

        def _feeds_through(node: str, link_id: str, arriving: bool) -> bool:
            """Whether traffic can continue through *node* other than by
            doubling back along this link's own carriageway pair.

            The subtlety that this exists for: on a two-way road both
            carriageways meet at the same terminal node, so that node has both
            an incoming and an outgoing corridor link and looks connected. It is
            still a gateway — the only way "onward" is a U-turn back down the
            road you came in on. Excluding the reverse of the link under test is
            what distinguishes a dead end from a junction.
            """
            from_node, to_node, _ = corridor[link_id]
            candidates = incoming[node] if arriving else outgoing[node]
            for other_id in candidates:
                if other_id == link_id:
                    continue
                other_from, other_to, _ = corridor[other_id]
                # The reverse carriageway of this same road: same node pair,
                # opposite orientation. Not a through movement.
                if (other_from, other_to) == (to_node, from_node):
                    continue
                return True
            return False

        termini: List[_Terminus] = []
        for link_id, (from_node, to_node, capacity) in corridor.items():
            # Inbound: nothing feeds this link's upstream node, so traffic on it
            # must have originated outside the modelled network.
            if not _feeds_through(from_node, link_id, arriving=True):
                x, y = nodes[from_node]
                termini.append(self._make_terminus(
                    link_id, x, y, INBOUND, capacity, envelope))
            # Outbound: nothing continues past this link's downstream node.
            if not _feeds_through(to_node, link_id, arriving=False):
                x, y = nodes[to_node]
                termini.append(self._make_terminus(
                    link_id, x, y, OUTBOUND, capacity, envelope))

        kept = [t for t in termini if t.peripherality >= self.min_peripherality]
        logger.debug(
            f"Termini: {len(termini)} raw, {len(kept)} past the peripherality "
            f"filter ({len(termini) - len(kept)} interior breaks discarded)"
        )
        return kept

    @staticmethod
    def _make_terminus(link_id: str, x: float, y: float, direction: str,
                       capacity: float,
                       envelope: DirectionalEnvelope) -> _Terminus:
        return _Terminus(
            link_id=link_id,
            x=x,
            y=y,
            direction=direction,
            capacity=capacity,
            peripherality=envelope.peripherality(x, y),
            compass=envelope.compass(x, y),
        )

    def _deduplicate(self, termini: Sequence[_Terminus]) -> List[Cordon]:
        """Collapse termini into distinct gateways.

        Two termini belong to the same gateway when they carry traffic the same
        way and their crossing points are within ``dedupe_radius_m``. Opposite
        directions never merge: an E->I trip must start on an inbound link and an
        I->E trip must end on an outbound one, so a divided motorway is properly
        two cordons, one per carriageway.

        Without this the weighting double-counts the biggest roads, because a
        motorway contributes a terminus per carriageway and per segment that
        falls at the boundary.
        """
        clusters: List[List[_Terminus]] = []

        # Largest roads first, so each cluster is seeded by its main link and the
        # resulting cordon id names the corridor's biggest segment.
        ordered = sorted(termini, key=lambda t: (-t.capacity, t.link_id))

        for terminus in ordered:
            placed = False
            for cluster in clusters:
                seed = cluster[0]
                if seed.direction != terminus.direction:
                    continue
                if math.hypot(seed.x - terminus.x,
                              seed.y - terminus.y) > self.dedupe_radius_m:
                    continue
                cluster.append(terminus)
                placed = True
                break
            if not placed:
                clusters.append([terminus])

        cordons: List[Cordon] = []
        for cluster in clusters:
            seed = cluster[0]
            cordons.append(Cordon(
                cordon_id=f"cordon_{seed.direction[:3]}_{seed.link_id}",
                x=seed.x,
                y=seed.y,
                direction=seed.direction,
                compass=seed.compass,
                link_ids=sorted({t.link_id for t in cluster}),
                # Capacity of the merged carriageway, not of every segment that
                # happened to fall in the cluster: summing collinear segments of
                # the same road would inflate a long corridor over a short one.
                capacity=max(t.capacity for t in cluster),
                peripherality=seed.peripherality,
            ))

        cordons.sort(key=lambda c: c.cordon_id)
        return cordons


def detector_from_config(config: Dict) -> CordonDetector:
    """Build a detector from the ``freight.cordon`` config block.

    Kept separate from the class so CordonDetector stays independent of the
    config schema and can be constructed directly in tests.
    """
    cordon_config = config.get('freight', {}).get('cordon', {})
    return CordonDetector(
        min_freespeed_ms=cordon_config.get('min_freespeed_ms', 22.0),
        min_capacity_vph=cordon_config.get('min_capacity_vph', 1500.0),
        min_peripherality=cordon_config.get('min_peripherality', 0.6),
        dedupe_radius_m=cordon_config.get('dedupe_radius_m', 1500.0),
        envelope_bins=cordon_config.get('envelope_bins', 36),
        envelope_percentile=cordon_config.get('envelope_percentile', 99.0),
        fail_if_none_found=cordon_config.get('fail_if_none_found', True),
    )


# How a cordon's sampling weight is chosen. See assign_cordon_weights.
#   hpms_truck_aadt — observed truck AADT where a cordon matched an HPMS
#                     segment, capacity elsewhere. The default, and the only
#                     value that uses observation at all.
#   capacity        — capacity for every cordon, ignoring observations. An
#                     ablation control: it answers "how much of the spatial
#                     distribution is data doing?" by removing the data.
WEIGHT_BY_HPMS_TRUCK_AADT = 'hpms_truck_aadt'
WEIGHT_BY_CAPACITY = 'capacity'
VALID_WEIGHT_BY = (WEIGHT_BY_HPMS_TRUCK_AADT, WEIGHT_BY_CAPACITY)


def weight_by_from_config(config: Dict) -> str:
    """Read and validate ``freight.cordon.weight_by``.

    Validated rather than silently defaulted: a misspelled value here would
    change the spatial distribution of every truck while the run still reported
    success, which is the class of failure this module is built to avoid.
    """
    cordon_config = config.get('freight', {}).get('cordon', {})
    value = cordon_config.get('weight_by', WEIGHT_BY_HPMS_TRUCK_AADT)
    if value not in VALID_WEIGHT_BY:
        raise ValueError(
            f"freight.cordon.weight_by must be one of {VALID_WEIGHT_BY}, "
            f"got {value!r}"
        )
    return value
