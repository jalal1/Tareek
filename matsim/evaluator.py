"""
MATSim Simulation Evaluator
Compares simulation outputs with ground truth traffic counts
"""

import gzip
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
from shapely import STRtree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as mpatches

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimulationEvaluator:
    """Evaluate MATSim simulation against ground truth traffic counts"""

    # Google-Maps-style traffic colormap: green → yellow → orange → dark red
    TRAFFIC_COLORS = ['#2ECC40', '#FFDC00', '#FF851B', "#FF0000"]

    # Congestion thresholds (speed/freespeed ratio boundaries)
    # [severe/congested, congested/moderate, moderate/free_flow]
    CONGESTION_THRESHOLDS = [0.3, 0.4, 0.7]
    CONGESTION_LABELS = ['Severe', 'Congested', 'Moderate', 'Free flow']

    def __init__(self, experiment_dir: Path, ground_truth_data_dir: Optional[Path] = None):
        """
        Initialize evaluator

        Args:
            experiment_dir: Path to experiment directory (e.g., experiments/experiment_20251231_170636)
            ground_truth_data_dir: Path to ground truth data directory (default: data/evaluation)
        """
        self.experiment_dir = Path(experiment_dir)
        self.evaluation_dir = self.experiment_dir / "evaluation"
        self.evaluation_dir.mkdir(exist_ok=True, parents=True)

        # Data paths
        if ground_truth_data_dir is None:
            self.data_dir = Path("../data/evaluation")
        else:
            self.data_dir = Path(ground_truth_data_dir)

        self.ground_truth_path = self.data_dir / "2024_hourly_volumes.csv"
        # Legacy TCDS file — may not exist for non-Twin-Cities regions
        self.locations_path = self.data_dir / "TCDS_locs_info.csv"

        # Network and linkstats will be set dynamically
        self.network_path = None
        self.linkstats_path = None

        # Cached data
        self.network_links = None
        self.link_geometries = None
        self._reverse_node_index = None  # {(to_node, from_node): link_id}
        self._link_strtree = None        # STRtree for spatial antiparallel search
        self._strtree_link_ids = None    # link_id list aligned with STRtree

        # Scaling factors for capacity-reduced simulations
        self.flow_capacity_factor = None
        self.storage_capacity_factor = None
        self.population_scaling_factor = None

        # County boundary polygons in UTM (loaded lazily)
        self._county_polygons_utm = None

    def load_scaling_factors(self) -> Tuple[float, float]:
        """
        Load scaling factors from experiment config.
        Uses flowCapacityFactor for volume scaling when comparing against
        real-world counts, as it reflects the true simulated-to-real traffic ratio.
        Also loads population scaling_factor for reference.

        Returns:
            Tuple of (flow_capacity_factor, storage_capacity_factor)
            Returns (1.0, 1.0) if config not found or factors not specified
        """
        # Check for config_used.json in experiment directory
        config_path = self.experiment_dir / 'config_used.json'

        if not config_path.exists():
            # Try alternative location
            config_path = self.experiment_dir / 'config.json'

        if not config_path.exists():
            logger.warning(f"No config file found in experiment directory, assuming scaling factors = 1.0")
            return 1.0, 1.0

        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get configurable params
            configurable_params = config.get('matsim', {}).get('configurable_params', {})

            flow_factor = configurable_params.get('qsim.flowCapacityFactor', 1.0)
            storage_factor = configurable_params.get('qsim.storageCapacityFactor', 1.0)

            # Get population scaling factor (the actual sample fraction)
            pop_scaling = config.get('plan_generation', {}).get('scaling_factor', 1.0)
            self.population_scaling_factor = float(pop_scaling)

            if abs(float(flow_factor) - float(pop_scaling)) > 0.01 and float(pop_scaling) < 1.0:
                logger.warning(
                    f"scaling_factor ({pop_scaling}) differs from flowCapacityFactor ({flow_factor}). "
                    f"Volume scaling uses flowCapacityFactor={flow_factor} — verify this matches "
                    f"the actual population sample fraction for correct extrapolation."
                )

            logger.info(f"Loaded scaling factors: population={pop_scaling}, flow={flow_factor}, storage={storage_factor}")
            return float(flow_factor), float(storage_factor)

        except Exception as e:
            logger.warning(f"Could not load scaling factors from config: {e}")
            return 1.0, 1.0

    def find_linkstats_file(self) -> Optional[Path]:
        """
        Automatically find the linkstats file in the experiment output directory.
        Looks for linkstats.txt.gz files in ITERS subdirectories.

        Returns:
            Path to linkstats file, or None if not found
        """
        output_dir = self.experiment_dir / "output"
        if not output_dir.exists():
            return None

        # Look for ITERS directory
        iters_dir = output_dir / "ITERS"
        if not iters_dir.exists():
            return None

        # Find all linkstats files
        linkstats_files = list(iters_dir.glob("it.*/*.linkstats.txt.gz"))

        if not linkstats_files:
            return None

        # Sort by iteration number and return the last one
        def get_iteration_num(path):
            try:
                return int(path.parent.name.split('.')[-1])
            except (ValueError, IndexError):
                return -1

        linkstats_files.sort(key=get_iteration_num)
        return linkstats_files[-1]

    def find_network_file(self) -> Optional[Path]:
        """
        Automatically find the network file in the experiment output directory.

        Returns:
            Path to network file, or None if not found
        """
        output_dir = self.experiment_dir / "output"
        if not output_dir.exists():
            return None

        # Check for both .xml.gz and .xml versions
        network_gz = output_dir / "output_network.xml.gz"
        network_xml = output_dir / "output_network.xml"

        if network_gz.exists():
            return network_gz
        elif network_xml.exists():
            return network_xml

        return None

    def load_device_locations(self) -> pd.DataFrame:
        """
        Load device location information with lat/lon coordinates

        Returns:
            DataFrame with columns: Loc ID, Latitude, Longitude
        """
        df = pd.read_csv(self.locations_path, encoding='utf-8-sig')

        # Rename for consistency
        df = df.rename(columns={'Loc ID': 'LOCAL_ID'})

        return df[['LOCAL_ID', 'Latitude', 'Longitude']]

    def load_network(self, network_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """
        Load MATSim network for visualization purposes.

        Args:
            network_path: Path to output_network.xml or output_network.xml.gz

        Returns:
            Tuple of (links_df, link_geometries)
        """
        self.network_path = Path(network_path)

        # Parse XML (handle both .xml and .xml.gz)
        if str(self.network_path).endswith('.gz'):
            with gzip.open(self.network_path, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(self.network_path)

        root = tree.getroot()

        # Extract nodes
        nodes = {}
        for node in root.findall('.//node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            nodes[node_id] = (x, y)

        # Extract links
        links_data = []
        link_geometries = {}

        for link in root.findall('.//link'):
            link_id = link.get('id')
            from_node = link.get('from')
            to_node = link.get('to')

            if from_node in nodes and to_node in nodes:
                from_x, from_y = nodes[from_node]
                to_x, to_y = nodes[to_node]

                # Extract OSM highway type from nested attributes
                highway_type = ''
                attrs_elem = link.find('attributes')
                if attrs_elem is not None:
                    for attr in attrs_elem.findall('attribute'):
                        if attr.get('name') == 'osm:way:highway':
                            highway_type = attr.text or ''
                            break

                links_data.append({
                    'link_id': link_id,
                    'from_node': from_node,
                    'to_node': to_node,
                    'from_x': from_x,
                    'from_y': from_y,
                    'to_x': to_x,
                    'to_y': to_y,
                    'highway_type': highway_type,
                })

                # Create LineString geometry
                link_geometries[link_id] = LineString([(from_x, from_y), (to_x, to_y)])

        links_df = pd.DataFrame(links_data)

        self.network_links = links_df
        self.link_geometries = link_geometries

        # Build reverse-node index: (from_node, to_node) → link_id
        # For a link A→B, the reverse link is B→A.
        self._reverse_node_index = {}
        for _, row in links_df.iterrows():
            key = (row['from_node'], row['to_node'])
            self._reverse_node_index[key] = row['link_id']

        # Build STRtree for spatial antiparallel search (divided highways)
        geom_list = []
        id_list = []
        for lid, geom in link_geometries.items():
            geom_list.append(geom)
            id_list.append(lid)
        self._link_strtree = STRtree(geom_list)
        self._strtree_link_ids = id_list

        return links_df, link_geometries

    def get_reverse_link_id(self, link_id: str) -> Optional[str]:
        """
        Get the reverse direction link ID.

        Tries three strategies:
        1. Suffix-based: swap 'f' ↔ 'r' suffix (road-only network convention)
        2. Node-based: find a link with swapped from/to nodes (same-node
           bidirectional roads, e.g. surface streets in pt2matsim)
        3. Spatial antiparallel: find a nearby link (~50 m) pointing in the
           opposite direction (divided highways / dual carriageways where each
           direction uses different OSM nodes)

        Args:
            link_id: The link ID

        Returns:
            The reverse link ID if it exists in the network, None otherwise
        """
        if link_id is None:
            return None

        link_id = str(link_id)

        # Strategy 1: suffix-based (backward compat with road-only networks)
        if link_id.endswith('f'):
            reverse_id = link_id[:-1] + 'r'
        elif link_id.endswith('r'):
            reverse_id = link_id[:-1] + 'f'
        else:
            reverse_id = None

        if reverse_id and self.link_geometries is not None and reverse_id in self.link_geometries:
            return reverse_id

        # Strategy 2: node-based reverse lookup (pt2matsim / generic networks)
        if self._reverse_node_index is not None and self.network_links is not None:
            row = self.network_links[self.network_links['link_id'] == link_id]
            if not row.empty:
                from_node = row.iloc[0]['from_node']
                to_node = row.iloc[0]['to_node']
                candidate = self._reverse_node_index.get((to_node, from_node))
                if candidate and candidate != link_id:
                    return candidate

        # Strategy 3: spatial antiparallel search (divided highways)
        if self._link_strtree is not None and self.network_links is not None:
            return self._find_antiparallel_link(link_id)

        return None

    def _find_antiparallel_link(self, link_id: str,
                                buffer_m: float = 50.0,
                                cos_threshold: float = -0.7) -> Optional[str]:
        """
        Find a nearby link pointing in the opposite direction.

        Used for divided highways where the two carriageways have separate
        OSM nodes.  Searches within *buffer_m* metres of the link midpoint
        and returns the closest antiparallel link (cosine of direction
        vectors < *cos_threshold*, i.e. within ~45° of opposite).
        """
        row = self.network_links[self.network_links['link_id'] == link_id]
        if row.empty:
            return None
        row = row.iloc[0]

        # Direction vector of the primary link
        dx = row['to_x'] - row['from_x']
        dy = row['to_y'] - row['from_y']
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return None
        ux, uy = dx / length, dy / length

        # Search around the link midpoint
        mx = (row['from_x'] + row['to_x']) / 2.0
        my = (row['from_y'] + row['to_y']) / 2.0
        search_geom = Point(mx, my).buffer(buffer_m)

        nearby_idx = self._link_strtree.query(search_geom)

        best_id = None
        best_dist = float('inf')
        midpoint = Point(mx, my)

        for idx in nearby_idx:
            cand_id = self._strtree_link_ids[idx]
            if cand_id == link_id:
                continue

            cand_row = self.network_links[self.network_links['link_id'] == cand_id]
            if cand_row.empty:
                continue
            cand_row = cand_row.iloc[0]

            # Direction vector of candidate
            cdx = cand_row['to_x'] - cand_row['from_x']
            cdy = cand_row['to_y'] - cand_row['from_y']
            clen = np.sqrt(cdx * cdx + cdy * cdy)
            if clen < 1e-6:
                continue

            # Cosine similarity (< -0.7 ≈ within ~45° of opposite)
            cos_sim = (ux * cdx + uy * cdy) / clen
            if cos_sim > cos_threshold:
                continue

            # Distance from primary midpoint to candidate geometry
            cand_geom = self.link_geometries[cand_id]
            dist = midpoint.distance(cand_geom)
            if dist < best_dist:
                best_dist = dist
                best_id = cand_id

        return best_id

    def load_linkstats(self, linkstats_path: Path, scale_volumes: bool = True) -> pd.DataFrame:
        """
        Load linkstats and extract average hourly volumes.
        Scales volumes by 1/flowCapacityFactor if scaling is enabled.

        Args:
            linkstats_path: Path to linkstats.txt.gz file
            scale_volumes: Whether to scale volumes by 1/flowCapacityFactor (default: True)

        Returns:
            DataFrame with columns: link_id, HRS0-1avg, HRS1-2avg, ..., HRS23-24avg
        """
        self.linkstats_path = Path(linkstats_path)

        # Read gzipped file
        with gzip.open(self.linkstats_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')

        # Extract only avg columns for each hour
        avg_cols = ['LINK'] + [f'HRS{i}-{i+1}avg' for i in range(24)]
        df_avg = df[avg_cols].copy()
        df_avg = df_avg.rename(columns={'LINK': 'link_id'})
        df_avg['link_id'] = df_avg['link_id'].astype(str)

        # Scale volumes using flowCapacityFactor to extrapolate to real-world volumes.
        # flowCapacityFactor reflects the true simulated-to-real traffic ratio,
        # which may differ from scaling_factor (e.g., 20% plans but 15% flow capacity
        # when the population sample underrepresents actual demand).
        scaling_factor = self.flow_capacity_factor
        if scale_volumes and scaling_factor is not None and scaling_factor > 0 and scaling_factor < 1.0:
            hour_cols = [col for col in df_avg.columns if col.startswith('HRS')]
            scaling_multiplier = 1.0 / scaling_factor
            df_avg[hour_cols] = df_avg[hour_cols] * scaling_multiplier
            logger.info(f"Scaled simulated volumes by {scaling_multiplier:.2f}x (1/{scaling_factor} population scaling)")

        return df_avg

    @staticmethod
    def _normalize_link_id(raw_id) -> Optional[str]:
        """Normalize a link ID that may have been mangled by CSV round-tripping.

        Pandas reads purely-numeric strings from CSV as float64, so a link ID
        written as "12345" comes back as 12345.0.  ``str(12345.0)`` produces
        ``"12345.0"`` which won't match the original ``"12345"`` in linkstats.
        This helper strips the spurious ``.0`` suffix when appropriate.
        """
        if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
            return None
        s = str(raw_id).strip()
        if not s:
            return None
        # "12345.0" → "12345"  (but leave "12345f" or "12345.67" alone)
        if s.endswith('.0'):
            try:
                return str(int(float(s)))
            except (ValueError, OverflowError):
                pass
        return s

    def compare_volumes(self, matched_devices: pd.DataFrame,
                       linkstats: pd.DataFrame) -> pd.DataFrame:
        """
        Compare ground truth volumes with simulation volumes.
        Sums volumes from both forward and reverse links when available,
        since traffic counting devices typically measure both directions.

        Args:
            matched_devices: Devices matched to links with ground truth
            linkstats: Simulation linkstats with hourly volumes

        Returns:
            DataFrame with comparison results for each device and hour
        """
        results = []
        zero_observed_count = 0
        bidirectional_device_count = 0
        unidirectional_device_count = 0

        # Validate hour columns exist in ground truth
        hour_cols_gt = [f'H{i:02d}' for i in range(1, 25)]

        # Create a set of link_ids for faster lookup
        linkstats_link_ids = set(linkstats['link_id'])

        for _, device in matched_devices.iterrows():
            device_id = device['LOCAL_ID']
            link_id = self._normalize_link_id(device['matched_link_id'])
            if not link_id:
                continue

            # Check for missing hour columns in this device's data
            missing_cols = [c for c in hour_cols_gt if c not in device.index]
            if missing_cols:
                logger.warning(f"Device {device_id} missing hour columns: {missing_cols}")
                continue

            # Get simulation data for the primary link
            sim_data = linkstats[linkstats['link_id'] == link_id]
            has_primary = not sim_data.empty

            if has_primary:
                sim_data = sim_data.iloc[0]

            # Check for reverse link — prefer the value from matched_devices.csv
            # (computed at counts-generation time), fall back to own lookup
            reverse_link_id = None
            if 'reverse_link_id' in device.index:
                reverse_link_id = self._normalize_link_id(device['reverse_link_id'])
            if not reverse_link_id:
                reverse_link_id = self.get_reverse_link_id(link_id)

            reverse_sim_data = None
            has_reverse = False

            if reverse_link_id and reverse_link_id in linkstats_link_ids:
                reverse_sim_data = linkstats[linkstats['link_id'] == reverse_link_id]
                if not reverse_sim_data.empty:
                    reverse_sim_data = reverse_sim_data.iloc[0]
                    has_reverse = True

            if has_reverse:
                bidirectional_device_count += 1
            else:
                unidirectional_device_count += 1

            # Compare each hour
            for hour in range(24):
                ground_truth_col = f'H{hour+1:02d}'
                sim_col = f'HRS{hour}-{hour+1}avg'

                observed = device[ground_truth_col]

                # Sum volumes from both directions (forward + reverse)
                simulated = 0.0
                if has_primary:
                    simulated += sim_data[sim_col]
                if has_reverse:
                    simulated += reverse_sim_data[sim_col]

                # Calculate metrics
                error = simulated - observed
                abs_error = abs(error)

                # Return np.nan for pct_error when observed=0 to exclude from statistics
                if observed > 0:
                    pct_error = (error / observed * 100)
                else:
                    pct_error = np.nan
                    zero_observed_count += 1

                # GEH statistic (standard for traffic validation)
                # Use NaN when both are zero — not a meaningful comparison
                if (simulated + observed) > 0:
                    geh = np.sqrt(2 * (simulated - observed)**2 / (simulated + observed))
                else:
                    geh = np.nan

                results.append({
                    'device_id': device_id,
                    'link_id': link_id,
                    'reverse_link_id': reverse_link_id if has_reverse else None,
                    'bidirectional': has_reverse,
                    'hour': hour,
                    'observed': observed,
                    'simulated': simulated,
                    'error': error,
                    'abs_error': abs_error,
                    'pct_error': pct_error,
                    'geh': geh
                })

        # Log matching summary
        total_devices = bidirectional_device_count + unidirectional_device_count
        logger.info(f"Volume comparison: {bidirectional_device_count}/{total_devices} devices "
                    f"matched bidirectionally (forward + reverse links)")
        if unidirectional_device_count > 0 and bidirectional_device_count == 0:
            logger.warning("No bidirectional matches found — simulated volumes may be ~50% of "
                          "observed if ground truth is bidirectional. Check network link ID format.")

        if zero_observed_count > 0:
            logger.info(f"Found {zero_observed_count} hourly observations with zero observed volume (pct_error set to NaN)")

        return pd.DataFrame(results)

    def calculate_summary_metrics(self, comparison_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the comparison

        Args:
            comparison_df: Comparison results from compare_volumes()

        Returns:
            Dictionary with summary metrics
        """
        if len(comparison_df) == 0:
            return {
                'num_devices': 0,
                'num_comparisons': 0,
                'mean_error': 0,
                'mae': 0,
                'rmse': 0,
                'mean_pct_error': 0,
                'mean_geh': 0,
                'geh_lt_5_pct': 0,
                'correlation': 0,
                'experiment_name': self.experiment_dir.name
            }

        # Count observations with zero observed volume (where pct_error is NaN)
        zero_observed_count = comparison_df['pct_error'].isna().sum()
        valid_pct_error_count = comparison_df['pct_error'].notna().sum()

        # Count bidirectional matches
        bidirectional_count = comparison_df['bidirectional'].sum() if 'bidirectional' in comparison_df.columns else 0
        unique_bidirectional_devices = comparison_df[comparison_df['bidirectional'] == True]['device_id'].nunique() if 'bidirectional' in comparison_df.columns else 0

        # GEH metrics excluding NaN (zero-zero pairs)
        valid_geh = comparison_df['geh'].dropna()
        geh_valid_count = len(valid_geh)

        # Correlation excluding zero-observed hours (consistent with pct_error)
        nonzero_obs = comparison_df[comparison_df['observed'] > 0]
        correlation = nonzero_obs[['observed', 'simulated']].corr().iloc[0, 1] if len(nonzero_obs) > 1 else 0.0

        # Peak-hour correlation (AM: 6-9, PM: 15-18)
        peak_hours = comparison_df[comparison_df['hour'].isin([6, 7, 8, 15, 16, 17])]
        peak_correlation = peak_hours[['observed', 'simulated']].corr().iloc[0, 1] if len(peak_hours) > 1 else 0.0

        metrics = {
            'num_devices': comparison_df['device_id'].nunique(),
            'num_comparisons': len(comparison_df),
            'num_bidirectional_devices': int(unique_bidirectional_devices),
            'mean_error': comparison_df['error'].mean(),
            'mae': comparison_df['abs_error'].mean(),
            'rmse': np.sqrt((comparison_df['error']**2).mean()),
            'mean_pct_error': comparison_df['pct_error'].mean(),  # NaN excluded automatically by pandas
            'pct_error_valid_count': int(valid_pct_error_count),
            'zero_observed_count': int(zero_observed_count),
            'mean_geh': valid_geh.mean() if geh_valid_count > 0 else 0.0,
            'geh_lt_5_pct': (valid_geh < 5).sum() / geh_valid_count * 100 if geh_valid_count > 0 else 0.0,
            'geh_valid_count': int(geh_valid_count),
            'correlation': correlation,
            'peak_hour_correlation': peak_correlation,
            'experiment_name': self.experiment_dir.name
        }

        return metrics

    def plot_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """
        Create visualization plots for the comparison

        Args:
            comparison_df: Comparison results
            save: Whether to save plots to evaluation directory
        """
        if len(comparison_df) == 0:
            logger.warning("Cannot generate plots: no comparison data available")
            return None

        # Create figure with overall title
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'Simulation Evaluation - {self.experiment_dir.name}',
                     fontsize=14, fontweight='bold', y=0.995)

        # Create subplots with adjusted spacing for suptitle
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(top=0.96)

        # 1. Scatter plot: Observed vs Simulated (peak hours colored)
        ax = axes[0, 0]
        peak_mask = comparison_df['hour'].isin([6, 7, 8, 15, 16, 17])
        off_peak = comparison_df[~peak_mask]
        peak = comparison_df[peak_mask]

        ax.scatter(off_peak['observed'], off_peak['simulated'],
                  alpha=0.2, s=8, c='#AAAAAA', label='Off-peak', zorder=2)
        ax.scatter(peak['observed'], peak['simulated'],
                  alpha=0.5, s=14, c='#E74C3C', label='Peak (6-9, 15-18)', zorder=3)

        max_val = max(comparison_df['observed'].max(), comparison_df['simulated'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect match', zorder=1)

        # Compute and annotate correlations
        nonzero_obs = comparison_df[comparison_df['observed'] > 0]
        corr_all = nonzero_obs[['observed', 'simulated']].corr().iloc[0, 1] if len(nonzero_obs) > 1 else 0
        corr_peak = peak[['observed', 'simulated']].corr().iloc[0, 1] if len(peak) > 1 else 0
        ax.text(0.05, 0.92, f'r (all)  = {corr_all:.3f}\nr (peak) = {corr_peak:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Observed Volume')
        ax.set_ylabel('Simulated Volume')
        ax.set_title('Observed vs Simulated Traffic Volumes')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        # 2. Hourly pattern
        ax = axes[0, 1]
        hourly_avg = comparison_df.groupby('hour')[['observed', 'simulated']].mean()
        ax.plot(hourly_avg.index, hourly_avg['observed'], 'o-', label='Observed')
        ax.plot(hourly_avg.index, hourly_avg['simulated'], 's-', label='Simulated')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Volume')
        ax.set_title('Average Hourly Traffic Pattern')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))

        # 3. GEH distribution (excluding NaN from zero-zero pairs)
        ax = axes[1, 0]
        valid_geh = comparison_df['geh'].dropna()
        ax.hist(valid_geh, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=5, color='r', linestyle='--', label='GEH = 5 (threshold)')
        ax.set_xlabel('GEH Statistic')
        ax.set_ylabel('Frequency')
        ax.set_title('GEH Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Error distribution (excluding NaN from zero-observed hours)
        ax = axes[1, 1]
        ax.hist(comparison_df['pct_error'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Percent Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Percentage Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plot_path = self.evaluation_dir / 'volume_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_path}")

        return fig

    def plot_spatial_maps(self, matched_devices: pd.DataFrame, linkstats: pd.DataFrame,
                          comparison_df: pd.DataFrame, save: bool = True):
        """
        Create spatial overview map showing network traffic

        Args:
            matched_devices: Devices matched to links with ground truth
            linkstats: Simulation linkstats
            comparison_df: Comparison results
            save: Whether to save plots to evaluation directory
        """
        if self.network_links is None or self.link_geometries is None:
            logger.warning("Network not loaded, cannot generate spatial maps")
            return None

        if len(matched_devices) == 0:
            logger.warning("No matched devices, cannot generate spatial maps")
            return None

        if comparison_df is None or len(comparison_df) == 0:
            logger.warning("No comparison data available, cannot generate spatial maps")
            return None

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_title(f'Spatial Overview - {self.experiment_dir.name}',
                     fontsize=16, fontweight='bold')

        # Prepare link data
        # Get total simulated volume for each link (sum across all hours)
        hour_cols = [f'HRS{i}-{i+1}avg' for i in range(24)]
        linkstats_with_total = linkstats.copy()
        linkstats_with_total['total_volume'] = linkstats[hour_cols].sum(axis=1)

        # Merge network links with linkstats
        links_with_volume = self.network_links.merge(
            linkstats_with_total[['link_id', 'total_volume']],
            on='link_id',
            how='left'
        )
        links_with_volume['total_volume'] = links_with_volume['total_volume'].fillna(0)

        # --- GEH device markers on gray network ---
        self._plot_traffic_heatmap(ax, links_with_volume, matched_devices, comparison_df)

        if save:
            plot_path = self.evaluation_dir / 'spatial_overview.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            logger.info(f"Saved spatial overview to {plot_path}")

        return fig

    def _plot_traffic_heatmap(self, ax, links_with_volume: pd.DataFrame,
                              matched_devices: pd.DataFrame, comparison_df: pd.DataFrame):
        """Plot network as gray background with GEH-colored device markers."""

        # Draw all links as gray background
        segments = [[(link['from_x'], link['from_y']), (link['to_x'], link['to_y'])]
                    for _, link in links_with_volume.iterrows()]
        lc = LineCollection(segments, colors='#9D9C9C', linewidths=0.8, alpha=0.5)
        ax.add_collection(lc)

        # Draw county boundary polygons
        self._draw_county_boundaries(ax)

        # Plot devices colored by GEH
        device_geh = comparison_df.groupby('device_id')['geh'].mean().reset_index()
        devices_with_geh = matched_devices.merge(device_geh, left_on='LOCAL_ID', right_on='device_id', how='left')

        # Color devices by GEH performance
        for _, device in devices_with_geh.iterrows():
            geh = device.get('geh', float('inf'))
            if geh < 5:
                color = '#2ECC40'  # Green: good
                marker = 'o'
            elif geh < 10:
                color = '#FFDC00'  # Yellow: acceptable
                marker = 's'
            else:
                color = '#FF4136'  # Red: poor
                marker = '^'

            ax.scatter(device['utm_x'], device['utm_y'], c=color, s=100,
                      edgecolors='black', linewidths=1.5, marker=marker, zorder=5)

        ax.set_aspect('equal')
        ax.set_xlabel('UTM X (m)')
        ax.set_ylabel('UTM Y (m)')
        ax.grid(True, alpha=0.3)

        # Legend below the map
        legend_elements = [
            mpatches.Patch(facecolor='#2ECC40', label='GEH < 5 (Good)', edgecolor='black'),
            mpatches.Patch(facecolor='#FFDC00', label='GEH 5-10 (Acceptable)', edgecolor='black'),
            mpatches.Patch(facecolor='#FF4136', label='GEH > 10 (Poor)', edgecolor='black'),
            mpatches.Patch(facecolor='none', edgecolor='blue', linestyle='--',
                           linewidth=2, label='County Boundary'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    def _plot_traffic_heatmap_clean(self, ax, links_with_volume: pd.DataFrame):
        """Plot network as gray background (no count stations or GEH)."""

        # Draw all links as gray
        segments = [[(link['from_x'], link['from_y']), (link['to_x'], link['to_y'])]
                    for _, link in links_with_volume.iterrows()]
        lc = LineCollection(segments, colors='#9D9C9C', linewidths=0.8, alpha=0.5)
        ax.add_collection(lc)

        self._draw_county_boundaries(ax)

        ax.set_aspect('equal')
        ax.set_xlabel('UTM X (m)')
        ax.set_ylabel('UTM Y (m)')
        ax.grid(True, alpha=0.3)

    def plot_heatmap_only(self, linkstats: pd.DataFrame, save: bool = True):
        """
        Create heatmap-only spatial map (no count stations or GEH).

        Args:
            linkstats: Simulation linkstats
            save: Whether to save to evaluation directory
        """
        if self.network_links is None:
            logger.warning("Network not loaded, cannot generate heatmap")
            return None

        hour_cols = [f'HRS{i}-{i+1}avg' for i in range(24)]
        linkstats_with_total = linkstats.copy()
        linkstats_with_total['total_volume'] = linkstats[hour_cols].sum(axis=1)

        links_with_volume = self.network_links.merge(
            linkstats_with_total[['link_id', 'total_volume']],
            on='link_id', how='left'
        )
        links_with_volume['total_volume'] = links_with_volume['total_volume'].fillna(0)

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_title(f'Network Overview - {self.experiment_dir.name}',
                     fontsize=16, fontweight='bold')

        self._plot_traffic_heatmap_clean(ax, links_with_volume)

        if save:
            plot_path = self.evaluation_dir / 'heatmap_daily.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            logger.info(f"Saved heatmap to {plot_path}")
            plt.close(fig)

        return fig

    @staticmethod
    def _smooth_congestion(hw_data: pd.DataFrame, method: str = 'neighbor') -> pd.DataFrame:
        """
        Smooth congestion ratios to reduce visual noise from short links.

        Args:
            hw_data: Highway data with 'congestion_ratio', 'LENGTH',
                     'from_node', 'to_node', 'from_x', 'from_y', 'to_x', 'to_y'
            method: 'neighbor' for length-weighted network neighbor averaging,
                    'gaussian' for spatial Gaussian kernel smoothing

        Returns:
            hw_data with smoothed 'congestion_ratio' column
        """
        has_ratio_mask = hw_data['congestion_ratio'].notna()
        if has_ratio_mask.sum() <= 1:
            return hw_data

        if method == 'neighbor':
            # Length-weighted average of each link with its network neighbors.
            # Short links get dominated by longer neighbors.
            node_to_links = defaultdict(list)
            for idx, row in hw_data[has_ratio_mask].iterrows():
                node_to_links[row['from_node']].append(idx)
                node_to_links[row['to_node']].append(idx)

            smoothed = hw_data['congestion_ratio'].copy()
            lengths = hw_data['LENGTH'].fillna(0)
            ratios_raw = hw_data['congestion_ratio']

            for idx in hw_data[has_ratio_mask].index:
                from_node = hw_data.loc[idx, 'from_node']
                to_node = hw_data.loc[idx, 'to_node']
                neighbor_idxs = set(node_to_links[from_node]) | set(node_to_links[to_node])
                total_weight = 0.0
                weighted_sum = 0.0
                for n_idx in neighbor_idxs:
                    r = ratios_raw.loc[n_idx]
                    if np.isnan(r):
                        continue
                    w = lengths.loc[n_idx]
                    weighted_sum += r * w
                    total_weight += w
                if total_weight > 0:
                    smoothed.loc[idx] = weighted_sum / total_weight

            hw_data['congestion_ratio'] = smoothed

        elif method == 'gaussian':
            # Gaussian kernel smoothing based on spatial distance between link midpoints.
            # Each link's ratio is a distance-weighted average of nearby links,
            # with weights decaying by exp(-d^2 / (2 * sigma^2)).
            # Sigma is set to the median link length to adapt to network scale.
            rated = hw_data[has_ratio_mask]
            mid_x = (rated['from_x'].values + rated['to_x'].values) / 2
            mid_y = (rated['from_y'].values + rated['to_y'].values) / 2
            ratios_arr = rated['congestion_ratio'].values
            rated_idx = rated.index.values

            median_length = hw_data.loc[has_ratio_mask, 'LENGTH'].median()
            sigma = max(median_length, 50.0)  # at least 50m
            two_sigma_sq = 2.0 * sigma * sigma
            # Only consider links within 3*sigma for efficiency
            cutoff = 3.0 * sigma
            cutoff_sq = cutoff * cutoff

            smoothed = hw_data['congestion_ratio'].copy()

            for i in range(len(rated_idx)):
                dx = mid_x - mid_x[i]
                dy = mid_y - mid_y[i]
                dist_sq = dx * dx + dy * dy
                within = dist_sq <= cutoff_sq
                weights = np.exp(-dist_sq[within] / two_sigma_sq)
                smoothed.loc[rated_idx[i]] = np.average(ratios_arr[within], weights=weights)

            hw_data['congestion_ratio'] = smoothed

        return hw_data

    def _load_linkstats_raw(self) -> Optional[pd.DataFrame]:
        """Load raw linkstats with LENGTH, FREESPEED, and TRAVELTIME columns."""
        if self.linkstats_path is None:
            return None
        with gzip.open(self.linkstats_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
        df = df.rename(columns={'LINK': 'link_id'})
        df['link_id'] = df['link_id'].astype(str)
        return df

    @staticmethod
    def _offset_segments(df: pd.DataFrame, offset_m: float) -> list:
        """
        Build line segments offset perpendicular to link direction.

        Each link is shifted to the right of its travel direction by offset_m meters,
        so forward/reverse links on the same road appear as separate lanes.
        """
        dx = df['to_x'].values - df['from_x'].values
        dy = df['to_y'].values - df['from_y'].values
        lengths = np.sqrt(dx**2 + dy**2)
        lengths = np.where(lengths == 0, 1.0, lengths)  # avoid division by zero
        # perpendicular unit vector (right-hand side of travel direction)
        perp_x = dy / lengths * offset_m
        perp_y = -dx / lengths * offset_m
        segments = [
            [(row['from_x'] + px, row['from_y'] + py),
             (row['to_x'] + px, row['to_y'] + py)]
            for (_, row), px, py in zip(df.iterrows(), perp_x, perp_y)
        ]
        return segments

    def plot_peak_hour_highway_heatmaps(self, linkstats: pd.DataFrame, save: bool = True,
                                           show_county_borders: bool = False, smoothing: str = 'gaussian'):
        """
        Create Google-Maps-style congestion heatmaps for 8 AM and 5 PM, highways only.

        Colors links by speed/freespeed ratio (congestion), not raw volume.
        Thresholds are defined in CONGESTION_THRESHOLDS class constant.

        Args:
            linkstats: Simulation linkstats (used for fallback; raw file re-read for speed data)
            save: Whether to save to evaluation directory
            show_county_borders: Whether to draw county boundary polygons
            smoothing: Smoothing method - 'neighbor' (length-weighted network averaging),
                       'gaussian' (spatial Gaussian kernel), or None to disable
        """
        if self.network_links is None:
            logger.warning("Network not loaded, cannot generate peak hour heatmaps")
            return None

        # Load raw linkstats to get LENGTH, FREESPEED, TRAVELTIME columns
        raw_linkstats = self._load_linkstats_raw()
        if raw_linkstats is None:
            logger.warning("Linkstats path not set, cannot generate peak hour heatmaps")
            return None

        highway_types = {'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link'} # remove 'secondary', 'secondary_link'
        is_highway = self.network_links['highway_type'].isin(highway_types)
        highway_links = self.network_links[is_highway]
        other_links = self.network_links[~is_highway]

        if len(highway_links) == 0:
            logger.warning("No highway links found in network, skipping peak hour heatmaps")
            return None

        logger.info(f"  Highway links for peak heatmaps: {len(highway_links):,} "
                    f"(of {len(self.network_links):,} total)")

        # hour_index 8 = 8:00-9:00 AM, hour_index 17 = 5:00-6:00 PM
        peak_hours = [
            (8, '8 AM', 'heatmap_8am_highways.png'),
            (17, '5 PM', 'heatmap_5pm_highways.png'),
        ]

        # Discrete colormap: each color maps to a threshold bin
        # TRAFFIC_COLORS = [green, yellow, orange, dark_red] → reverse for low=red, high=green
        t = self.CONGESTION_THRESHOLDS
        cmap = ListedColormap(list(reversed(self.TRAFFIC_COLORS)))
        bounds = [0] + t + [1.0]
        norm = BoundaryNorm(bounds, cmap.N)

        figs = []
        for hour_idx, label, filename in peak_hours:
            vol_col = f'HRS{hour_idx}-{hour_idx+1}avg'
            tt_col = f'TRAVELTIME{hour_idx}-{hour_idx+1}avg'

            # Merge highway links with raw linkstats
            merge_cols = ['link_id', 'LENGTH', 'FREESPEED']
            if vol_col in raw_linkstats.columns:
                merge_cols.append(vol_col)
            if tt_col in raw_linkstats.columns:
                merge_cols.append(tt_col)

            hw_data = highway_links.merge(
                raw_linkstats[merge_cols],
                on='link_id', how='left'
            )

            # Compute congestion ratio: actual_speed / freespeed
            has_traffic = hw_data[vol_col].fillna(0) > 0 if vol_col in hw_data.columns else pd.Series(False, index=hw_data.index)
            has_traveltime = hw_data[tt_col].fillna(0) > 0 if tt_col in hw_data.columns else pd.Series(False, index=hw_data.index)
            can_compute = has_traffic & has_traveltime & (hw_data['FREESPEED'].fillna(0) > 0)

            # actual_speed = LENGTH / TRAVELTIME, ratio = actual_speed / FREESPEED
            hw_data['congestion_ratio'] = np.nan
            if can_compute.any():
                actual_speed = hw_data.loc[can_compute, 'LENGTH'] / hw_data.loc[can_compute, tt_col]
                hw_data.loc[can_compute, 'congestion_ratio'] = np.minimum(
                    actual_speed / hw_data.loc[can_compute, 'FREESPEED'], 1.0
                )

            # Smooth congestion ratios to reduce visual noise from short links
            if smoothing:
                hw_data = self._smooth_congestion(hw_data, method=smoothing)

            fig, ax = plt.subplots(figsize=(14, 12))
            ax.set_title(f'Highway Congestion at {label} - {self.experiment_dir.name}',
                         fontsize=16, fontweight='bold')

            # Draw all non-highway links as gray background
            bg_segments = []
            for _, link in other_links.iterrows():
                bg_segments.append([(link['from_x'], link['from_y']),
                                    (link['to_x'], link['to_y'])])
            if bg_segments:
                bg_lc = LineCollection(bg_segments, colors="#9D9C9C",
                                      linewidths=0.5, alpha=0.4, zorder=1)
                ax.add_collection(bg_lc)

            # Split highways into: has congestion data vs no traffic (drawn gray)
            has_ratio = hw_data['congestion_ratio'].notna()

            # Offset distance in meters so forward/reverse links appear as separate lanes
            lane_offset = 50

            # Links with no traffic in this hour → thin gray
            no_traffic = hw_data[~has_ratio]
            if len(no_traffic) > 0:
                nt_segments = self._offset_segments(no_traffic, lane_offset)
                nt_lc = LineCollection(nt_segments, colors="#CCCCCC",
                                      linewidths=1.0, alpha=0.5, zorder=2)
                ax.add_collection(nt_lc)

            # Links with traffic → color by congestion ratio
            with_ratio = hw_data[has_ratio]
            if len(with_ratio) > 0:
                hw_segments = self._offset_segments(with_ratio, lane_offset)
                ratios = with_ratio['congestion_ratio'].values

                hw_lc = LineCollection(hw_segments, cmap=cmap, norm=norm,
                                       linewidths=1.5, alpha=0.85, zorder=3)
                hw_lc.set_array(ratios)
                ax.add_collection(hw_lc)

            if show_county_borders:
                self._draw_county_boundaries(ax)

            ax.set_aspect('equal')
            ax.set_xlabel('UTM X (m)')
            ax.set_ylabel('UTM Y (m)')
            ax.grid(True, alpha=0.3)

            ax.autoscale_view()

            # Vertical colorbar showing congestion ratio
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Speed / Free-flow Speed', rotation=270, labelpad=20)
            cbar.set_ticks([0] + t + [1.0])
            cbar.set_ticklabels(['0 (gridlock)'] + [str(v) for v in t] + ['1.0 (free flow)'])

            # Log congestion stats
            if len(with_ratio) > 0:
                r = ratios
                labels = self.CONGESTION_LABELS
                counts = [
                    (r < t[0]).sum(),
                    ((r >= t[0]) & (r < t[1])).sum(),
                    ((r >= t[1]) & (r < t[2])).sum(),
                    (r >= t[2]).sum(),
                ]
                logger.info(f"  {label} congestion — " +
                            ", ".join(f"{labels[i]} ({counts[i]:,})" for i in range(4)) +
                            f", no traffic: {len(no_traffic):,}")

            if save:
                plot_path = self.evaluation_dir / filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                logger.info(f"Saved peak hour heatmap to {plot_path}")
                plt.close(fig)

            figs.append(fig)

        return figs

    def _load_county_polygons_utm(self) -> List:
        """
        Load county boundary polygons from Census TIGER shapefile, projected to UTM.

        Reads region.counties (FIPS GEOIDs) and coordinates.utm_epsg from
        config_used.json. Downloads the shapefile on first use.

        Returns:
            List of Shapely polygons in UTM coordinates. Empty list on failure.
        """
        if self._county_polygons_utm is not None:
            return self._county_polygons_utm

        self._county_polygons_utm = []

        config_path = self.experiment_dir / 'config_used.json'
        if not config_path.exists():
            logger.warning("config_used.json not found, cannot load county boundaries")
            return self._county_polygons_utm

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            county_geoids = config.get('region', {}).get('counties', [])
            utm_epsg = config.get('coordinates', {}).get('utm_epsg')
            data_dir = config.get('data', {}).get('data_dir', '')

            if not county_geoids or not utm_epsg or not data_dir:
                logger.warning("Missing region.counties, coordinates.utm_epsg, "
                               "or data.data_dir in config")
                return self._county_polygons_utm

            from utils.region_utils import load_county_polygons
            self._county_polygons_utm = load_county_polygons(
                county_geoids, data_dir, utm_epsg=utm_epsg
            )

        except Exception as e:
            logger.warning(f"Failed to load county polygons: {e}")

        return self._county_polygons_utm

    def _draw_county_boundaries(self, ax):
        """
        Draw dashed county boundary polygons on the spatial overview map.

        Falls back silently if county polygons cannot be loaded.
        """
        polygons = self._load_county_polygons_utm()
        if not polygons:
            return

        for poly in polygons:
            if hasattr(poly, 'geoms'):
                # MultiPolygon — draw each part
                parts = list(poly.geoms)
            else:
                parts = [poly]

            for part in parts:
                xs, ys = part.exterior.coords.xy
                ax.plot(xs, ys, color='blue', linestyle='--',
                        linewidth=2, zorder=4)

    def generate_per_device_reports(self, matched_devices: pd.DataFrame,
                                     comparison_df: pd.DataFrame, linkstats: pd.DataFrame):
        """
        Generate individual reports for each device

        Args:
            matched_devices: Devices matched to links
            comparison_df: Comparison results
            linkstats: Simulation linkstats
        """
        if len(matched_devices) == 0:
            logger.warning("No matched devices, cannot generate per-device reports")
            return

        # Create device_reports subdirectory
        reports_dir = self.evaluation_dir / 'device_reports'
        reports_dir.mkdir(exist_ok=True)

        logger.info(f"Generating per-device reports for {len(matched_devices)} devices...")

        for _, device in matched_devices.iterrows():
            device_id = device['LOCAL_ID']
            self._generate_single_device_report(device, comparison_df, linkstats, reports_dir)

        logger.info(f"Per-device reports saved to {reports_dir}")

    def _generate_single_device_report(self, device: pd.Series, comparison_df: pd.DataFrame,
                                       linkstats: pd.DataFrame, reports_dir: Path):
        """Generate report for a single device"""

        device_id = device['LOCAL_ID']
        link_id = self._normalize_link_id(device['matched_link_id']) or ''
        device_data = comparison_df[comparison_df['device_id'] == device_id]

        if len(device_data) == 0:
            return

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], hspace=0.4, wspace=0.3)

        ax_map = fig.add_subplot(gs[0, :])  # Top: location map
        ax_time = fig.add_subplot(gs[1, :])  # Middle: time series
        ax_geh = fig.add_subplot(gs[2, 0])   # Bottom left: GEH values
        ax_stats = fig.add_subplot(gs[2, 1]) # Bottom right: stats table

        # 1. Location map (simplified - just show device and link)
        self._plot_device_location_map(ax_map, device, link_id)

        # 2. 24-hour traffic comparison
        self._plot_device_time_series(ax_time, device_data, device_id)

        # 3. Hourly GEH values
        self._plot_device_geh(ax_geh, device_data)

        # 4. Summary statistics
        self._plot_device_stats(ax_stats, device_data)

        # Overall title
        fig.suptitle(f'Device {device_id} - Validation Report', fontsize=16, fontweight='bold', y=0.98)

        # Save with link_id in filename
        report_path = reports_dir / f'device_{device_id}_linkid_{link_id}.png'
        plt.savefig(report_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_device_location_map(self, ax, device: pd.Series, link_id: str):
        """Plot device location on map"""
        # Get matched link geometry
        link_geom = self.link_geometries.get(link_id)

        # Collect all points to determine bounds
        all_x = [device['utm_x']]
        all_y = [device['utm_y']]

        if link_geom:
            # Plot link
            x, y = link_geom.xy
            ax.plot(x, y, 'b-', linewidth=3, label='Matched Link', zorder=1)
            all_x.extend(x)
            all_y.extend(y)

            # Plot link midpoint
            link_midpoint = link_geom.interpolate(0.5, normalized=True)
            ax.scatter(link_midpoint.x, link_midpoint.y, c='blue', s=150, marker='o',
                      edgecolors='white', linewidths=2, label='Link Midpoint', zorder=3)

        # Plot device
        ax.scatter(device['utm_x'], device['utm_y'], c='red', s=200, marker='*',
                  edgecolors='black', linewidths=2, label='Device', zorder=4)

        # Draw connection line to midpoint
        if link_geom:
            link_midpoint = link_geom.interpolate(0.5, normalized=True)
            ax.plot([device['utm_x'], link_midpoint.x], [device['utm_y'], link_midpoint.y],
                   'k--', linewidth=1.5, alpha=0.6, label='Distance', zorder=0)

        # Calculate bounds and ensure square aspect ratio with padding
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Calculate the range in each dimension
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Use the larger range as the base, with a minimum size of 100m
        max_range = max(x_range, y_range, 100)

        # Add 20% padding on each side
        padding = max_range * 0.2
        padded_range = max_range + 2 * padding

        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Set equal limits centered on the data
        ax.set_xlim(x_center - padded_range / 2, x_center + padded_range / 2)
        ax.set_ylim(y_center - padded_range / 2, y_center + padded_range / 2)

        ax.set_aspect('equal')
        ax.set_title(f"Device Location (Link: {link_id}, Distance to midpoint: {device['distance_m']:.1f}m)",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('UTM X (m)', fontsize=10)
        ax.set_ylabel('UTM Y (m)', fontsize=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, borderaxespad=0)
        ax.grid(True, alpha=0.3)

    def _plot_device_time_series(self, ax, device_data: pd.DataFrame, device_id):
        """Plot 24-hour comparison"""
        # Sort by hour
        device_data = device_data.sort_values('hour')

        ax.plot(device_data['hour'], device_data['observed'], 'o-', linewidth=2,
               markersize=6, label='Observed (Ground Truth)', color='#2ECC40')
        ax.plot(device_data['hour'], device_data['simulated'], 's-', linewidth=2,
               markersize=6, label='Simulated', color='#FF4136')

        ax.set_ylabel('Traffic Volume (vehicles/hour)', fontsize=11)
        ax.set_title('24-Hour Traffic Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([])  # Hide x-axis labels (shared with GEH plot below)

    def _plot_device_geh(self, ax, device_data: pd.DataFrame):
        """Plot hourly GEH values"""
        device_data = device_data.sort_values('hour')

        colors = ['#2ECC40' if geh < 5 else '#FFDC00' if geh < 10 else '#FF4136'
                 for geh in device_data['geh']]

        ax.bar(device_data['hour'], device_data['geh'], color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='GEH = 5 (threshold)')
        ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='GEH = 10')

        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('GEH Statistic', fontsize=11)
        ax.set_title('Hourly GEH Values', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 24, 2))

    def _plot_device_stats(self, ax, device_data: pd.DataFrame):
        """Plot summary statistics as table"""
        ax.axis('off')

        # Calculate stats
        mae = device_data['abs_error'].mean()
        rmse = np.sqrt((device_data['error']**2).mean())
        mean_geh = device_data['geh'].mean()
        geh_lt_5_pct = (device_data['geh'] < 5).sum() / len(device_data) * 100
        correlation = device_data[['observed', 'simulated']].corr().iloc[0, 1]

        # Create table
        stats_data = [
            ['Metric', 'Value'],
            ['Mean Abs Error (MAE)', f'{mae:.1f} veh/hr'],
            ['Root Mean Sq Error (RMSE)', f'{rmse:.1f} veh/hr'],
            ['Mean GEH', f'{mean_geh:.2f}'],
            ['Hours with GEH < 5', f'{geh_lt_5_pct:.1f}%'],
            ['Correlation', f'{correlation:.3f}'],
        ]

        table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style alternating rows
        for i in range(1, len(stats_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')

        # No title — avoids overlap with the table header row

    def run_evaluation(self, network_path: Optional[Path] = None,
                      linkstats_path: Optional[Path] = None,
                      generate_spatial_maps: bool = True,
                      generate_per_device_reports: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete evaluation pipeline.

        Requires matched_devices.csv to exist in the experiment directory (generated by counts_generator).
        If not found, evaluation cannot proceed.

        Args:
            network_path: Path to output_network.xml (optional, will auto-detect if not provided)
            linkstats_path: Path to linkstats.txt.gz (optional, will auto-detect if not provided)
            generate_spatial_maps: Whether to generate spatial overview maps (default: True)
            generate_per_device_reports: Whether to generate per-device reports (default: False)

        Returns:
            Tuple of (comparison_df, summary_metrics)
        """
        # Check for matched_devices.csv (generated by counts_generator)
        matched_devices_path = self.experiment_dir / 'matched_devices.csv'
        if not matched_devices_path.exists():
            logger.warning(f"matched_devices.csv not found in {self.experiment_dir}")
            logger.warning("This file is generated during counts.xml creation.")
            logger.warning("If counts generation was disabled or failed, evaluation cannot proceed.")
            # Return empty results
            return pd.DataFrame(), {
                'num_devices': 0,
                'num_comparisons': 0,
                'error': 'matched_devices.csv not found',
                'experiment_name': self.experiment_dir.name
            }

        # Load matched devices
        logger.info(f"Loading matched devices from: {matched_devices_path}")
        matched_devices = pd.read_csv(matched_devices_path)
        logger.info(f"Loaded {len(matched_devices)} devices matched to network links")

        if len(matched_devices) == 0:
            logger.warning("No devices in matched_devices.csv!")
            logger.warning("Evaluation cannot proceed without matched devices.")
            return pd.DataFrame(), {
                'num_devices': 0,
                'num_comparisons': 0,
                'error': 'no matched devices',
                'experiment_name': self.experiment_dir.name
            }

        # Auto-detect files if not provided
        if network_path is None:
            network_path = self.find_network_file()
            if network_path is None:
                raise FileNotFoundError(f"Could not find network file in {self.experiment_dir / 'output'}")
            logger.info(f"Auto-detected network file: {network_path.relative_to(self.experiment_dir)}")

        if linkstats_path is None:
            linkstats_path = self.find_linkstats_file()
            if linkstats_path is None:
                raise FileNotFoundError(f"Could not find linkstats file in {self.experiment_dir / 'output'}")
            logger.info(f"Auto-detected linkstats file: {linkstats_path.relative_to(self.experiment_dir)}")

        # Load scaling factors from experiment config
        logger.info("Loading scaling factors from experiment config...")
        self.flow_capacity_factor, self.storage_capacity_factor = self.load_scaling_factors()

        # Load network for spatial maps
        logger.info("Loading network for spatial visualization...")
        self.load_network(network_path)
        logger.info(f"Loaded {len(self.network_links):,} links")

        logger.info("Loading linkstats...")
        linkstats = self.load_linkstats(linkstats_path)
        logger.info(f"Loaded {len(linkstats):,} links from linkstats")

        logger.info("Comparing volumes...")
        comparison_df = self.compare_volumes(matched_devices, linkstats)
        logger.info(f"Generated {len(comparison_df):,} hourly comparisons")

        logger.info("Calculating summary metrics...")
        summary_metrics = self.calculate_summary_metrics(comparison_df)

        # Add network extent info to metrics
        try:
            summary_metrics['total_devices_all_regions'] = len(self.load_device_locations())
        except (FileNotFoundError, Exception):
            # Device locations file may not exist for non-Twin-Cities regions
            summary_metrics['total_devices_all_regions'] = len(matched_devices)
        summary_metrics['devices_in_network_area'] = len(matched_devices)
        summary_metrics['network_links'] = len(self.network_links)

        # Add scaling factors to metrics
        summary_metrics['flow_capacity_factor'] = self.flow_capacity_factor
        summary_metrics['storage_capacity_factor'] = self.storage_capacity_factor
        summary_metrics['population_scaling_factor'] = self.population_scaling_factor
        scaling_factor = self.flow_capacity_factor
        if scaling_factor and scaling_factor < 1.0:
            summary_metrics['volume_scaling_applied'] = True
            summary_metrics['volume_scaling_multiplier'] = 1.0 / scaling_factor
        else:
            summary_metrics['volume_scaling_applied'] = False

        # Save results
        comparison_path = self.evaluation_dir / 'volume_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison results to {comparison_path}")

        metrics_path = self.evaluation_dir / 'summary_metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        logger.info(f"Saved summary metrics to {metrics_path}")

        logger.info("Generating plots...")
        self.plot_comparison(comparison_df)

        # Generate spatial maps if requested and we have matched devices
        if generate_spatial_maps and len(matched_devices) > 0:
            logger.info("Generating spatial overview maps...")
            self.plot_spatial_maps(matched_devices, linkstats, comparison_df)

        # Always generate heatmap-only and peak hour highway maps when spatial maps are requested
        if generate_spatial_maps:
            logger.info("Generating heatmap-only map...")
            self.plot_heatmap_only(linkstats)
            logger.info("Generating peak hour highway heatmaps...")
            self.plot_peak_hour_highway_heatmaps(linkstats)

        # Generate per-device reports if requested
        if generate_per_device_reports and len(matched_devices) > 0:
            logger.info("Generating per-device reports...")
            self.generate_per_device_reports(matched_devices, comparison_df, linkstats)

        return comparison_df, summary_metrics
