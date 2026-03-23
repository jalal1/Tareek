from collections import Counter
from utils.logger import setup_logger
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from data_sources.base_survey_trip import BaseSurveyTrip

logger = setup_logger(__name__)


def is_home_work_home_chain(chain_pattern: str) -> bool:
    """
    Check if chain matches: Home → ... → Work → ... → Home

    Args:
        chain_pattern: Chain pattern string (e.g., "Home-Shopping-Work-Dining-Home")

    Returns:
        True if chain starts with Home, contains Work, and ends with Home

    Examples:
        >>> is_home_work_home_chain("Home-Work-Home")
        True
        >>> is_home_work_home_chain("Home-Shopping-Work-Home")
        True
        >>> is_home_work_home_chain("Home-Shopping-Home")
        False
        >>> is_home_work_home_chain("Work-Home")
        False
    """
    activities = [a.strip() for a in chain_pattern.split('-')]

    if len(activities) < 3:
        return False

    return (activities[0] == BaseSurveyTrip.ACT_HOME and
            BaseSurveyTrip.ACT_WORK in activities and
            activities[-1] == BaseSurveyTrip.ACT_HOME)


def is_home_other_home_chain(chain_pattern: str) -> bool:
    """
    Check if chain matches: Home → ... → Home (without Work)

    Args:
        chain_pattern: Chain pattern string

    Returns:
        True if chain starts and ends with Home and does NOT contain Work

    Examples:
        >>> is_home_other_home_chain("Home-Shopping-Home")
        True
        >>> is_home_other_home_chain("Home-Dining-Shopping-Home")
        True
        >>> is_home_other_home_chain("Home-Work-Home")
        False
    """
    activities = [a.strip() for a in chain_pattern.split('-')]

    if len(activities) < 2:
        return False

    return (activities[0] == BaseSurveyTrip.ACT_HOME and
            BaseSurveyTrip.ACT_WORK not in activities and
            activities[-1] == BaseSurveyTrip.ACT_HOME)


def filter_chains_by_type(chains_df: pd.DataFrame,
                          chain_type: str = 'home_work_home') -> pd.DataFrame:
    """
    Filter chains by type for future extensibility.

    Supported types:
    - 'home_work_home': Went home → ... → Work → ... → Went home
    - 'home_other_home': Went home → ... → Went home (no Work)
    - 'all': Return all chains without filtering

    Args:
        chains_df: DataFrame with 'pattern' column containing chain patterns
        chain_type: Type of chains to filter

    Returns:
        Filtered DataFrame with only matching chains

    Raises:
        ValueError: If chain_type is not recognized
    """
    if chain_type == 'all':
        return chains_df

    def _contains_work(pattern: str) -> bool:
        """Check if chain contains Work activity (any structure)."""
        activities = [a.strip() for a in pattern.split('-')]
        return BaseSurveyTrip.ACT_WORK in activities and len(activities) >= 3

    filter_functions = {
        'home_work_home': is_home_work_home_chain,
        'home_other_home': is_home_other_home_chain,
        'contains_work': _contains_work,
    }

    if chain_type not in filter_functions:
        raise ValueError(f"Unknown chain type '{chain_type}'. "
                        f"Supported types: {list(filter_functions.keys())}")

    filter_func = filter_functions[chain_type]
    mask = chains_df['pattern'].apply(filter_func)
    filtered_df = chains_df[mask].copy()

    logger.info(f"Filtered chains by type '{chain_type}': "
               f"{len(filtered_df)}/{len(chains_df)} chains match")

    return filtered_df

def process_trip_chains(persons: Dict[str, Dict[str, pd.DataFrame]], use_weight: bool = False) -> List[Dict]:
    """Process trip chains from person trips dictionary, optionally using trip weights.

    Args:
        persons: Dictionary of person trips {person_id: {date: trips_df}}
        use_weight: Whether to use the 'trip_weight' column when counting chains.

    Returns:
        List of chain patterns with frequencies
    """
    try:
        logger.info('Starting trip chain processing...')

        chains = []
        # debug only:
        # for person_id, days in list(persons.items())[30:50]:
        for person_id, days in persons.items():
            # print(f"Processing person_id: {person_id}")
            for date, trips_df in days.items():
                # print("trips", trips_df)
                # Skip if no trips
                if trips_df.empty:
                    continue

                # Sort trips by departure time
                sorted_trips = trips_df.sort_values(BaseSurveyTrip.DEPART_TIME)

                # Use canonical column names
                o_purpose_col = BaseSurveyTrip.ORIGIN_PURPOSE
                d_purpose_col = BaseSurveyTrip.DESTINATION_PURPOSE

                # Build activity chain
                chain_activities = []
                chain_weights = []
                for _, trip in sorted_trips.iterrows():
                    if use_weight:
                        chain_weights.append(trip.get('trip_weight', 1))
                    if not chain_activities:
                        chain_activities.append(trip[o_purpose_col])
                    # if trip[d_purpose_col] != chain_activities[-1]:
                    # Always append destination (preserves consecutive identical destinations)
                    chain_activities.append(trip[d_purpose_col])

                # print("Chain activities:", chain_activities)
                # print("Chain weights:", chain_weights)
                # Compute chain weight: min weight if using weight, else 1
                # Example where the weight for some trips may be larger than others
                # Processing person_id: 2300136902
                # Chain activities: ['Shopping', 'Home', 'Social', 'Home']
                # Chain weights: [198.5975799560547, 198.5975799560547, 287.1282043457031, 198.5975799560547]
                # Chain weight: 287.1282043457031
                if use_weight:
                    valid_weights = [w for w in chain_weights if w > 0]
                    if valid_weights:
                        # Use mean weight: trip_weight represents how many people
                        # this respondent represents, NOT per-trip multiplier.
                        # Using sum would bias longer chains to appear more frequent.
                        chain_weight = sum(valid_weights) / len(valid_weights)
                    else:
                        chain_weight = 0
                else:
                    chain_weight = 1

                # print("Chain weight:", chain_weight)

                chains.append({
                    'person_id': person_id,
                    'date': date,
                    'chain': '-'.join(chain_activities),
                    'length': len(chain_activities),
                    'weight': chain_weight
                })

        # Calculate chain frequencies (weighted if requested)
        chain_counter = Counter()
        for chain in chains:
            chain_counter[chain['chain']] += chain['weight'] if use_weight else 1

        total_chains = sum(chain_counter.values())

        # Prepare records
        records = []
        for pattern, frequency in chain_counter.items():
            records.append({
                'pattern': pattern,
                'frequency': frequency,
                'probability': frequency / total_chains if total_chains > 0 else 0,
                'is_valid': True  # placeholder for validation logic
            })

        logger.info(f'Successfully processed {len(chains)} trip chains')
        return records

    except Exception as e:
        logger.error(f"Error processing trip chains: {str(e)}")
        raise


class TripChainModel:
    """
    Model to learn trip chain probabilities using 2nd-order Markov chains
    and sample new chains while constraining to realistic patterns.
    """

    def __init__(self, chains_df: pd.DataFrame, home_boost_factor: float = 2.0,
                 length_distribution_df: pd.DataFrame = None,
                 early_stop_exponent: float = 2.0):
        """
        Initialize the model with survey data.

        Args:
            chains_df: DataFrame with columns 'pattern', 'frequency', 'probability'
            home_boost_factor: Multiplier for Home starting probability to account for
                             implicit home starts not captured in survey (default: 2.0)
            length_distribution_df: Optional DataFrame with chain patterns to learn
                the chain length distribution from. When None, the length distribution
                is learned from chains_df (the filtered data). Pass the full unfiltered
                survey chains here so that the 'generated' sampling method targets
                realistic chain lengths rather than the shorter lengths typical of
                purpose-filtered subsets.
            early_stop_exponent: Controls how aggressively chains are stopped early.
                1.0 = linear (original), 2.0 = quadratic (default, gentler),
                3.0 = cubic. Higher values let chains grow closer to target_length.
        """
        self.chains_df = chains_df.copy()
        self.home_boost_factor = home_boost_factor
        self._length_distribution_df = length_distribution_df
        self._early_stop_exponent = early_stop_exponent
        self.chain_patterns = None
        self.chain_probabilities = None

        # Distributions
        self.start_activity_dist = None
        self.end_activity_dist = None
        self.chain_length_dist = None
        self.second_order_transitions = None  # P(A_t | A_{t-2}, A_{t-1})
        self.first_order_transitions = None   # P(A_t | A_{t-1}) - fallback
        self.unique_activities = None

        self._fit()

    def _fit(self):
        """Fit the model by extracting patterns and computing distributions."""
        # Store chain patterns and normalize probabilities
        self.chain_patterns = self.chains_df['pattern'].values
        self.chain_probabilities = self.chains_df['probability'].values / self.chains_df['probability'].sum()

        # Extract unique activities
        self._extract_unique_activities()

        # Extract start and end distributions
        self._extract_start_end_distributions()

        # Extract chain length distribution
        self._extract_length_distribution()

        # Extract 2nd and 1st order transition probabilities
        self._extract_transition_probabilities()

    def _extract_unique_activities(self):
        """Extract unique activities from all patterns, excluding specific ones."""
        exclude = {"Change mode", "Not imputable"}
        activities = set()
        for pattern in self.chain_patterns:
            for a in pattern.split('-'):
                activity = a.strip()
                if activity not in exclude:
                    activities.add(activity)
        self.unique_activities = sorted(activities)

    def _extract_start_end_distributions(self):
        """Extract probability distributions for starting and ending activities."""
        start_counts = defaultdict(float)
        end_counts = defaultdict(float)

        for pattern, probability in zip(self.chain_patterns, self.chain_probabilities):
            activities = [a.strip() for a in pattern.split('-')]

            # Count starts and ends
            start_activity = activities[0]
            end_activity = activities[-1]

            start_counts[start_activity] += probability
            end_counts[end_activity] += probability

        # Apply home boost factor to account for implicit home starts in survey
        if BaseSurveyTrip.ACT_HOME in start_counts:
            start_counts[BaseSurveyTrip.ACT_HOME] *= self.home_boost_factor

        # Normalize
        self.start_activity_dist = dict(sorted(start_counts.items()))
        self.start_activity_dist = {k: v / sum(start_counts.values()) for k, v in start_counts.items()}

        self.end_activity_dist = dict(sorted(end_counts.items()))
        self.end_activity_dist = {k: v / sum(end_counts.values()) for k, v in end_counts.items()}

    def _extract_length_distribution(self):
        """Extract the distribution of chain lengths.

        When ``length_distribution_df`` was supplied at construction time the
        length distribution is learned from that (the full, unfiltered survey)
        so that the Markov generator targets realistic chain lengths.
        Otherwise it falls back to the (possibly filtered) ``chains_df``.
        """
        source_df = self._length_distribution_df if self._length_distribution_df is not None else None

        if source_df is not None and not source_df.empty:
            # Use the external (unfiltered) length distribution
            patterns = source_df['pattern'].values
            probs = source_df['probability'].values
            probs = probs / probs.sum()  # re-normalize
        else:
            # Fallback: use own (filtered) patterns
            patterns = self.chain_patterns
            probs = self.chain_probabilities

        length_counts = defaultdict(float)
        for pattern, probability in zip(patterns, probs):
            activities = [a.strip() for a in pattern.split('-')]
            length = len(activities)
            length_counts[length] += probability

        # Normalize
        self.chain_length_dist = {k: v / sum(length_counts.values())
                                  for k, v in sorted(length_counts.items())}

    def _extract_transition_probabilities(self):
        """Extract 2nd-order and 1st-order transition probabilities."""
        # 2nd order: P(A_t | A_{t-2}, A_{t-1})
        second_order_counts = defaultdict(lambda: defaultdict(float))
        second_order_totals = defaultdict(float)

        # 1st order: P(A_t | A_{t-1}) - fallback when 2nd order not available
        first_order_counts = defaultdict(lambda: defaultdict(float))
        first_order_totals = defaultdict(float)

        for pattern, probability in zip(self.chain_patterns, self.chain_probabilities):
            activities = [a.strip() for a in pattern.split('-')]

            # Extract transitions
            for i in range(len(activities)):
                if i >= 2:
                    # 2nd order transition
                    prev_prev = activities[i-2]
                    prev = activities[i-1]
                    curr = activities[i]

                    key = (prev_prev, prev)
                    second_order_counts[key][curr] += probability
                    second_order_totals[key] += probability

                if i >= 1:
                    # 1st order transition
                    prev = activities[i-1]
                    curr = activities[i]

                    first_order_counts[prev][curr] += probability
                    first_order_totals[prev] += probability

        # Normalize 2nd order
        self.second_order_transitions = {}
        for key, transitions in second_order_counts.items():
            self.second_order_transitions[key] = {
                activity: count / second_order_totals[key]
                for activity, count in transitions.items()
            }

        # Normalize 1st order
        self.first_order_transitions = {}
        for activity, transitions in first_order_counts.items():
            self.first_order_transitions[activity] = {
                next_activity: count / first_order_totals[activity]
                for next_activity, count in transitions.items()
            }

    def sample_chain_length(self) -> int:
        """Sample a chain length from the learned distribution."""
        lengths = list(self.chain_length_dist.keys())
        probabilities = list(self.chain_length_dist.values())
        return np.random.choice(lengths, p=probabilities)

    def sample_direct_pattern(self) -> str:
        """Sample a complete chain pattern directly from observed data."""
        return np.random.choice(self.chain_patterns, p=self.chain_probabilities)

    def sample_generated_chain(self, max_length: int = None, min_length: int = 3) -> str:
        """
        Generate a new chain using 2nd-order Markov transitions
        constrained to realistic patterns.

        The chain length is controlled by two mechanisms:
        1. A target_length sampled from the survey distribution acts as the
           soft target — beyond it, the full end_prob kicks in.
        2. A hard ceiling (max_length or target+4) prevents runaway chains.

        Args:
            max_length: Hard ceiling on chain length. If None, uses
                        target_length + 4 (generous room to grow).
            min_length: Minimum chain length (default 3 = Home-X-Home minimum)

        Returns:
            Generated chain pattern as string
        """
        # Sample a target length from the survey distribution.
        # This is the soft target where the early stop reaches full strength.
        target_length = self.sample_chain_length()
        target_length = max(target_length, min_length)

        # Hard ceiling — if max_length is explicitly set, respect it;
        # otherwise use a generous default so chains can grow beyond target.
        if max_length is not None:
            hard_max = max(max_length, min_length)
        else:
            hard_max = max(target_length + 4, 10)

        # Sample starting activity
        start_activities = list(self.start_activity_dist.keys())
        start_probs = list(self.start_activity_dist.values())
        first_activity = np.random.choice(start_activities, p=start_probs)

        chain = [first_activity]

        # Build chain up to hard_max
        while len(chain) < hard_max:
            if len(chain) == 1:
                # Sample 2nd activity using 1st order transitions
                if first_activity in self.first_order_transitions:
                    next_activities = list(self.first_order_transitions[first_activity].keys())
                    next_probs = list(self.first_order_transitions[first_activity].values())
                    next_activity = np.random.choice(next_activities, p=next_probs)
                else:
                    next_activity = np.random.choice(self.unique_activities)
                chain.append(next_activity)
            else:
                # Use 2nd order transitions
                prev_prev = chain[-2]
                prev = chain[-1]
                key = (prev_prev, prev)

                if key in self.second_order_transitions:
                    next_activities = list(self.second_order_transitions[key].keys())
                    next_probs = list(self.second_order_transitions[key].values())
                    next_activity = np.random.choice(next_activities, p=next_probs)
                else:
                    # Fallback to 1st order
                    if prev in self.first_order_transitions:
                        next_activities = list(self.first_order_transitions[prev].keys())
                        next_probs = list(self.first_order_transitions[prev].values())
                        next_activity = np.random.choice(next_activities, p=next_probs)
                    else:
                        next_activity = np.random.choice(self.unique_activities)

                chain.append(next_activity)

                # Probabilistic early stop using survey-derived target length.
                # Below target_length: stop prob ramps up gently toward target.
                # At/beyond target_length: stop prob = full end_prob from survey.
                # The ramp shape is controlled by early_stop_exponent:
                #   1.0 = linear, 2.0 = quadratic (default, gentler ramp),
                #   3.0 = cubic (very gentle early, steep near target).
                if len(chain) >= min_length:
                    curr_activity = chain[-1]
                    end_prob = self.end_activity_dist.get(curr_activity, 0)

                    if len(chain) >= target_length:
                        # At or past target: full stop probability
                        scaled_end_prob = end_prob
                    else:
                        # Ramp up toward target
                        progress = (len(chain) - min_length) / max(1, target_length - min_length)
                        progress = progress ** self._early_stop_exponent
                        scaled_end_prob = end_prob * progress

                    if scaled_end_prob > 0 and np.random.random() < scaled_end_prob:
                        break

        return '-'.join(chain)

    def sample(self, method: str = 'direct',
               max_length: int = None, min_length: int = 3) -> str:
        """
        Sample a trip chain.

        Args:
            method: 'direct' to sample from observed patterns,
                    'generated' to create using Markov transitions
            max_length: Max activities in chain (None = sample from distribution)
            min_length: Min activities in chain

        Returns:
            Sampled chain pattern as string
        """
        if method == 'direct':
            return self.sample_direct_pattern()
        elif method == 'generated':
            return self.sample_generated_chain(
                max_length=max_length, min_length=min_length
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_samples(self, n_samples: int, method: str = 'direct',
                         max_length: int = None, min_length: int = 3) -> List[str]:
        """
        Generate multiple chain samples.

        Args:
            n_samples: Number of samples to generate
            method: 'direct' or 'generated'
            max_length: Max activities in chain (None = sample from distribution)
            min_length: Min activities in chain

        Returns:
            List of sampled chain patterns
        """
        return [self.sample(method=method, max_length=max_length,
                            min_length=min_length) for _ in range(n_samples)]

    def get_summary(self) -> dict:
        """Return summary statistics of the model."""
        return {
            'total_observed_patterns': len(self.chain_patterns),
            'unique_activities': self.unique_activities,
            'chain_length_distribution': dict(self.chain_length_dist),
            'start_activity_distribution': dict(self.start_activity_dist),
            'end_activity_distribution': dict(self.end_activity_dist),
            'num_2nd_order_transitions': len(self.second_order_transitions),
            'num_1st_order_transitions': len(self.first_order_transitions)
        }

    # Add this function to your TripChainModel class
    def visualize_generated_samples(self, n_samples: int = 1000, method: str = 'generated'):
        """
        Generate samples and visualize their distributions.

        Args:
            n_samples: Number of samples to generate
            method: 'direct' or 'generated'
        """
        # Generate samples
        samples = self.generate_samples(n_samples, method=method)

        # Extract chain lengths
        lengths = [len([a.strip() for a in s.split('-')]) for s in samples]

        # Extract starting activities
        starts = [s.split('-')[0].strip() for s in samples]

        # Extract ending activities
        ends = [s.split('-')[-1].strip() for s in samples]

        # Extract all activities and their frequencies
        all_activities = []
        for sample in samples:
            all_activities.extend([a.strip() for a in sample.split('-')])

        # Count occurrences
        length_counts = Counter(lengths)
        start_counts = Counter(starts)
        end_counts = Counter(ends)
        activity_counts = Counter(all_activities)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Generated Samples Distribution (n={n_samples}, method={method})',
                    fontsize=16, fontweight='bold')

        # 1. Chain length distribution
        ax = axes[0, 0]
        lengths_sorted = sorted(length_counts.keys())
        counts = [length_counts[l] for l in lengths_sorted]
        ax.bar(lengths_sorted, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Chain Length', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Chain Length Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (l, c) in enumerate(zip(lengths_sorted, counts)):
            ax.text(l, c + max(counts)*0.01, str(c), ha='center', fontsize=9)

        # 2. Starting activity distribution
        ax = axes[0, 1]
        starts_sorted = sorted(start_counts.items(), key=lambda x: x[1], reverse=True)
        start_labels, start_freqs = zip(*starts_sorted)
        start_probs = [f/sum(start_freqs)*100 for f in start_freqs]
        bars = ax.barh(start_labels, start_probs, color='coral', edgecolor='black')
        ax.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Starting Activity Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, prob) in enumerate(zip(bars, start_probs)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)

        # 3. Ending activity distribution
        ax = axes[1, 0]
        ends_sorted = sorted(end_counts.items(), key=lambda x: x[1], reverse=True)
        end_labels, end_freqs = zip(*ends_sorted)
        end_probs = [f/sum(end_freqs)*100 for f in end_freqs]
        bars = ax.barh(end_labels, end_probs, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Ending Activity Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, prob) in enumerate(zip(bars, end_probs)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)

        # 4. Overall activity frequency
        ax = axes[1, 1]
        activities_sorted = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
        activity_labels, activity_freqs = zip(*activities_sorted)
        activity_probs = [f/sum(activity_freqs)*100 for f in activity_freqs]
        bars = ax.barh(activity_labels, activity_probs, color='plum', edgecolor='black')
        ax.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Overall Activity Frequency', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, prob) in enumerate(zip(bars, activity_probs)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        # Log summary statistics
        logger.info("=" * 60)
        logger.info(f"GENERATED SAMPLES SUMMARY (n={n_samples}, method={method})")
        logger.info("=" * 60)
        logger.info(f"Average chain length: {np.mean(lengths):.2f}")
        logger.info(f"Min chain length: {min(lengths)}")
        logger.info(f"Max chain length: {max(lengths)}")
        logger.info(f"Unique starting activities: {len(start_counts)}")
        logger.info(f"Unique ending activities: {len(end_counts)}")
        logger.info(f"Unique activities overall: {len(activity_counts)}")
        logger.info(f"Unique chain patterns generated: {len(set(samples))}")


# # Example usage
# if __name__ == "__main__":
#     # Create sample dataframe
#     data = {
#         'pattern': ['Work-Went home', 'Went home-Work-Went home', 'School-Went home'],
#         'frequency': [245942.7595, 155975.6433, 144644.9827],
#         'probability': [0.024697257, 0.015662874, 0.014525064]
#     }
#     df = pd.DataFrame(data)

#     # Initialize and fit model
#     model = TripChainModel(df)

#     # Print summary
#     print("=" * 60)
#     print("MODEL SUMMARY")
#     print("=" * 60)
#     summary = model.get_summary()
#     for key, value in summary.items():
#         print(f"{key}: {value}")
#     print()

#     # Generate samples using direct method
#     print("=" * 60)
#     print("DIRECT METHOD (from observed patterns)")
#     print("=" * 60)
#     direct_samples = model.generate_samples(5, method='direct')
#     for i, sample in enumerate(direct_samples, 1):
#         print(f"{i}. {sample}")
#     print()

#     # Generate samples using generated method
#     print("=" * 60)
#     print("GENERATED METHOD (using 2nd-order Markov chains)")
#     print("=" * 60)
#     generated_samples = model.generate_samples(5, method='generated')
#     for i, sample in enumerate(generated_samples, 1):
#         print(f"{i}. {sample}")


class BlendedTripChainModel:
    """Thin wrapper that blends multiple per-source TripChainModels.

    At sampling time, a source is chosen with probability proportional to
    the configured survey weight.  The sample is then drawn from that
    source's TripChainModel.  This provides distribution-level blending
    without mixing the underlying Markov structures.

    When only one source is configured, callers should use the raw
    TripChainModel directly — this class is only needed for multi-source.
    """

    def __init__(self, models: Dict[str, TripChainModel],
                 weights: Dict[str, float]):
        """
        Args:
            models:  {source_key: TripChainModel} — one per survey.
            weights: {source_key: float} — raw weights from config
                     (``data.surveys[].weight``). Normalised internally.
        """
        self.models = models
        self.source_names = list(models.keys())

        raw = np.array([weights[name] for name in self.source_names],
                       dtype=float)
        total = raw.sum()
        if total <= 0:
            raise ValueError("Sum of blend weights must be positive")
        self.probabilities = raw / total

        logger.info(
            f"BlendedTripChainModel: {len(self.source_names)} sources — "
            + ", ".join(
                f"{name}: {prob:.2%}"
                for name, prob in zip(self.source_names, self.probabilities)
            )
        )

    # ── Public interface (matches TripChainModel) ─────────────────────

    def sample(self, method: str = 'direct',
               max_length: int = None, min_length: int = 3) -> str:
        """Sample one trip chain from a randomly chosen source."""
        source = np.random.choice(self.source_names, p=self.probabilities)
        return self.models[source].sample(
            method=method, max_length=max_length, min_length=min_length
        )

    def generate_samples(self, n_samples: int, method: str = 'direct',
                         max_length: int = None, min_length: int = 3) -> List[str]:
        """Generate *n_samples* trip chains via weighted source selection."""
        return [self.sample(method=method, max_length=max_length,
                            min_length=min_length) for _ in range(n_samples)]

    @property
    def chains_df(self) -> pd.DataFrame:
        """Return chains_df from the first source (used for diagnostics)."""
        return self.models[self.source_names[0]].chains_df
