"""
Candidate Scorer and Ranking Module.

Evaluates and ranks CDR3 candidates based on multiple criteria:
- Structure quality (pLDDT)
- Binding affinity (docking score)
- Sequence diversity
- Stability metrics
"""

import os
import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Amino acid properties
AA_PROPERTIES = {
    # Hydrophobicity (Kyte-Doolittle scale)
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# CDR3 preferred amino acids (from natural nanobody libraries)
CDR3_PREFERRED = {
    'preferred': set('ARYCDGFLMNPQSTV'),  # Common in CDR3
    'highly_preferred': set('ARY'),  # Very common
    'avoid': set('DEHIK'),  # Less common in CDR3
}

# Aromatic amino acids
AROMATIC = set('FWY')


@dataclass
class ScorerConfig:
    """Configuration for candidate scoring."""
    # Weights for final score
    weight_affinity: float = 0.3
    weight_plddt: float = 0.25
    weight_diversity: float = 0.15
    weight_stability: float = 0.15
    weight_docking: float = 0.15  # New: docking score weight

    # Thresholds
    min_plddt: float = 50.0
    max_plddt: float = 100.0
    min_cdr3_length: int = 8
    max_cdr3_length: int = 20

    # Diversity parameters
    min_unique_ratio: float = 0.7

    # Docking parameters
    use_docking: bool = False
    target_protein: str = "generic"


class CandidateScorer:
    """
    Scorer for CDR3 candidates.

    Computes comprehensive scores based on:
    1. Structure quality (pLDDT)
    2. Binding affinity estimate
    3. Sequence diversity
    4. Stability metrics
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        """Initialize scorer."""
        self.config = config or ScorerConfig()

    def score_candidate(
        self,
        candidate: Dict,
        structure_result: Optional[Dict] = None,
        all_candidates: Optional[List[Dict]] = None,
        docking_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Score a single candidate.

        Args:
            candidate: Candidate dictionary with 'sequence' key
            structure_result: Structure prediction results
            all_candidates: All candidates for diversity calculation
            docking_result: Docking results with binding energy

        Returns:
            Candidate with added scores
        """
        seq = candidate.get("sequence", "")

        # Initialize scores
        scores = {
            "sequence": seq,
            "length": len(seq),
        }

        # 1. Sequence quality score
        scores["sequence_score"] = self._compute_sequence_score(seq)

        # 2. Structure quality score (if available)
        if structure_result:
            scores["plddt_score"] = self._compute_plddt_score(structure_result)
            scores["structure_metrics"] = structure_result
        else:
            # Use heuristic
            scores["plddt_score"] = self._estimate_plddt_score(seq)

        # 3. Diversity score (computed later with all candidates)
        if all_candidates:
            scores["diversity_score"] = self._compute_diversity_score(seq, all_candidates)
        else:
            scores["diversity_score"] = 0.5

        # 4. Stability score
        scores["stability_score"] = self._compute_stability_score(seq)

        # 5. Affinity estimate (heuristic based on sequence properties)
        scores["affinity_score"] = self._estimate_affinity_score(seq)

        # 6. Docking score (if available)
        if docking_result:
            scores["docking_score"] = self._compute_docking_score(docking_result)
            scores["binding_energy"] = docking_result.get("binding_energy", 0)
            scores["estimated_kd"] = docking_result.get("affinity_nM", None)
        else:
            scores["docking_score"] = 0.5  # Neutral when no docking data

        # 7. Final weighted score
        final_score = (
            self.config.weight_affinity * scores["affinity_score"]
            + self.config.weight_plddt * scores["plddt_score"]
            + self.config.weight_diversity * scores["diversity_score"]
            + self.config.weight_stability * scores["stability_score"]
            + self.config.weight_docking * scores["docking_score"]
        )

        scores["final_score"] = final_score
        candidate.update(scores)

        return candidate

    def _compute_sequence_score(self, seq: str) -> float:
        """Compute sequence quality score based on amino acid composition."""
        if not seq:
            return 0.0

        score = 0.0

        # Length score (optimal CDR3 length is ~10-16)
        length = len(seq)
        if 10 <= length <= 16:
            length_score = 1.0
        elif 8 <= length <= 20:
            length_score = 0.7
        else:
            length_score = 0.3

        score += 0.3 * length_score

        # Preferred amino acid score
        preferred_count = sum(1 for aa in seq if aa in CDR3_PREFERRED['preferred'])
        avoid_count = sum(1 for aa in seq if aa in CDR3_PREFERRED['avoid'])

        pref_ratio = preferred_count / length if length > 0 else 0
        avoid_ratio = avoid_count / length if length > 0 else 0

        aa_score = pref_ratio - 0.5 * avoid_ratio
        score += 0.4 * max(0, min(1, aa_score))

        # Aromatic residue score (important for binding)
        aromatic_count = sum(1 for aa in seq if aa in AROMATIC)
        aromatic_ratio = aromatic_count / length if length > 0 else 0
        aromatic_score = min(1.0, aromatic_ratio * 3)  # Up to 33% aromatic is good

        score += 0.3 * aromatic_score

        return min(1.0, max(0.0, score))

    def _compute_plddt_score(self, structure_result: Dict) -> float:
        """Compute score from pLDDT values."""
        import math

        mean_plddt = structure_result.get("mean_plddt", 0)
        cdr3_plddt = structure_result.get("cdr3_plddt", 0)

        # Handle NaN case
        if cdr3_plddt is None or (isinstance(cdr3_plddt, float) and math.isnan(cdr3_plddt)):
            cdr3_plddt = mean_plddt  # Fallback to mean

        # Weight CDR3 region more heavily
        combined = 0.4 * mean_plddt + 0.6 * cdr3_plddt

        # Normalize to 0-1
        return min(1.0, max(0.0, (combined - self.config.min_plddt) /
                           (self.config.max_plddt - self.config.min_plddt)))

    def _estimate_plddt_score(self, seq: str) -> float:
        """Estimate pLDDT score from sequence (when no structure prediction)."""
        # Use sequence properties as proxy
        length = len(seq)

        # Longer sequences tend to have lower confidence
        if 8 <= length <= 16:
            length_factor = 1.0
        elif length < 8:
            length_factor = 0.8
        else:
            length_factor = max(0.5, 1.0 - (length - 16) * 0.05)

        # Preferred amino acids indicate better designability
        preferred = sum(1 for aa in seq if aa in CDR3_PREFERRED['preferred'])
        pref_ratio = preferred / length if length > 0 else 0

        # Combine
        score = 0.6 * length_factor + 0.4 * pref_ratio

        # Map to typical pLDDT range (50-85)
        estimated_plddt = 50 + score * 35
        return min(1.0, max(0.0, (estimated_plddt - self.config.min_plddt) /
                           (self.config.max_plddt - self.config.min_plddt)))

    def _compute_diversity_score(
        self,
        seq: str,
        all_candidates: List[Dict],
    ) -> float:
        """Compute diversity score based on uniqueness and pairwise distance."""
        if not all_candidates:
            return 0.5

        sequences = [c.get("sequence", "") for c in all_candidates]
        total = len(sequences)

        if total < 2:
            return 0.5

        # 1. Uniqueness ratio
        unique = len(set(sequences))
        uniqueness = unique / total

        # 2. Average Levenshtein distance
        distances = []
        for other_seq in sequences:
            if other_seq != seq:
                dist = self._levenshtein_distance(seq, other_seq)
                distances.append(dist)

        avg_dist = np.mean(distances) if distances else 0
        max_len = max(len(s) for s in sequences)
        normalized_dist = avg_dist / max_len if max_len > 0 else 0

        # Combine
        diversity_score = 0.5 * uniqueness + 0.5 * normalized_dist

        return min(1.0, max(0.0, diversity_score))

    def _compute_stability_score(self, seq: str) -> float:
        """Compute stability score based on sequence properties."""
        if not seq:
            return 0.0

        length = len(seq)

        # 1. Hydrophobicity score
        # CDR3 should have moderate hydrophobicity for stability
        hydro_scores = [AA_PROPERTIES.get(aa, 0) for aa in seq]
        mean_hydro = np.mean(hydro_scores)

        # Optimal hydrophobicity is around 0 (slightly hydrophilic)
        hydro_score = 1.0 - abs(mean_hydro) / 5.0
        hydro_score = max(0, min(1, hydro_score))

        # 2. Cysteine pairs (disulfide bonds enhance stability)
        cysteine_count = seq.count('C')
        if cysteine_count >= 2:
            # Check if they're spaced appropriately for disulfide
            cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
            has_disulfide = any(
                3 <= cys_positions[j] - cys_positions[i] <= 10
                for i in range(len(cys_positions))
                for j in range(i+1, len(cys_positions))
            )
            cys_score = 1.0 if has_disulfide else 0.6
        elif cysteine_count == 1:
            cys_score = 0.3
        else:
            cys_score = 0.5

        # 3. Proline content (rigid, can stabilize loops)
        proline_count = seq.count('P')
        proline_score = min(1.0, proline_count / 3)  # 1-3 prolines is good

        # 4. Glycine content (flexibility, avoid too many)
        glycine_count = seq.count('G')
        gly_score = max(0, 1.0 - glycine_count / 5)  # Few is good

        # Combine
        stability_score = (
            0.35 * hydro_score +
            0.25 * cys_score +
            0.20 * proline_score +
            0.20 * gly_score
        )

        return min(1.0, max(0.0, stability_score))

    def _estimate_affinity_score(self, seq: str) -> float:
        """Estimate binding affinity based on sequence features."""
        if not seq:
            return 0.0

        length = len(seq)
        score = 0.0

        # 1. Aromatic residues in CDR3 (important for binding)
        aromatic = sum(1 for aa in seq if aa in AROMATIC)
        aromatic_ratio = aromatic / length if length > 0 else 0

        # 2. Tyrosine content (very common in antigen-binding)
        tyrosine = seq.count('Y')
        tyr_score = min(1.0, tyrosine / 2)  # 1-2 tyrosines is good

        # 3. Arginine/Lysine (positively charged, may interact with antigens)
        positively_charged = seq.count('R') + seq.count('K')
        charge_score = min(1.0, positively_charged / 2)

        # 4. Diversity at key positions
        # Check variety of amino acids
        unique_aa = len(set(seq))
        diversity_score = min(1.0, unique_aa / 8)  # 8+ different AA is good

        # Combine with weights
        score = (
            0.30 * min(1.0, aromatic_ratio * 4) +
            0.30 * tyr_score +
            0.20 * charge_score +
            0.20 * diversity_score
        )

        return min(1.0, max(0.0, score))

    def _compute_docking_score(self, docking_result: Dict) -> float:
        """
        Compute docking score from docking results.

        Args:
            docking_result: Dict with 'binding_energy' (kcal/mol) and 'affinity_nM'

        Returns:
            Normalized score (0-1)
        """
        binding_energy = docking_result.get("binding_energy", 0)

        # Normalize binding energy to 0-1 score
        # Typical range: -3 to -10 kcal/mol
        # More negative = better binding
        # Map: -3 -> 0, -10 -> 1
        # So: score = (binding_energy - (-3)) / (-10 - (-3)) * (-1)
        # Or simpler: score = (-3 - binding_energy) / (-10 - (-3))
        #              = (-3 - binding_energy) / (-7)
        #              = (binding_energy + 3) / 7 (since both negated)
        #
        # For -3: score = 0
        # For -10: score = 1
        # For -6: score = (-6 + 3) / (-7) = -3/-7 = 0.43
        #
        # Actually, let's use: (min_energy - binding_energy) / (min_energy - max_energy)
        # where min_energy = -10, max_energy = -3
        # For -3: (−10 − (−3)) / (−10 − (−3)) = −7/−7 = 1 (wrong)
        #
        # Correct formula: (binding_energy - max_energy) / (min_energy - max_energy)
        # For -3: (-3 - (-3)) / (-10 - (-3)) = 0/−7 = 0
        # For -10: (-10 - (-3)) / (-10 - (-3)) = −7/−7 = 1
        # For -6: (-6 - (-3)) / (-10 - (-3)) = −3/−7 = 0.43
        min_energy = -10.0
        max_energy = -3.0

        score = (binding_energy - max_energy) / (min_energy - max_energy)
        return max(0.0, min(1.0, score))

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def rank_candidates(
        self,
        candidates: List[Dict],
        structure_results: Optional[Dict[str, Dict]] = None,
        docking_results: Optional[Dict[str, Dict]] = None,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Rank candidates by final score.

        Args:
            candidates: List of candidate dictionaries
            structure_results: Optional dict mapping sequence to structure results
            docking_results: Optional dict mapping sequence to docking results
            top_n: Number of top candidates to return

        Returns:
            Ranked list of candidates
        """
        # Score all candidates
        scored = []
        for i, candidate in enumerate(candidates):
            seq = candidate.get("sequence", "")
            struct_result = structure_results.get(seq) if structure_results else None
            dock_result = docking_results.get(seq) if docking_results else None

            scored_candidate = self.score_candidate(
                candidate,
                structure_result=struct_result,
                all_candidates=candidates,
                docking_result=dock_result,
            )
            scored_candidate["rank"] = i + 1
            scored.append(scored_candidate)

        # Sort by final score (descending)
        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Update ranks
        for i, candidate in enumerate(scored):
            candidate["rank"] = i + 1

        # Return top N
        return scored[:top_n]


class BindingEstimator:
    """
    Estimate binding affinity using heuristic scoring.

    This is a simplified version - real binding affinity requires
    molecular docking or experimental validation.
    """

    def __init__(self):
        """Initialize binding estimator."""
        # Common CDR3 motifs that interact with antigens
        self.binding_motifs = [
            "YRY", "YYG", "RYG", "YNY", "NYG",
            "ARL", "RLG", "LGR", "GRL",
            "RNY", "NYR", "YRN",
        ]

    def estimate_binding(
        self,
        cdr3_sequence: str,
        target_type: str = "generic",
    ) -> Dict:
        """
        Estimate binding affinity.

        Args:
            cdr3_sequence: CDR3 amino acid sequence
            target_type: Type of target (generic/her2/spike/etc)

        Returns:
            Dictionary with binding estimates
        """
        length = len(cdr3_sequence)
        score = 0.0

        # 1. Motif matching
        motif_matches = 0
        for motif in self.binding_motifs:
            if motif in cdr3_sequence:
                motif_matches += 1
        motif_score = min(1.0, motif_matches * 0.3)

        # 2. Sequence complexity
        unique_aa = len(set(cdr3_sequence))
        complexity_score = min(1.0, unique_aa / 10)

        # 3. Hydrophobic moments (simplified)
        hydro_score = self._compute_hydrophobic_moment(cdr3_sequence)

        # Combine
        binding_score = 0.4 * motif_score + 0.3 * complexity_score + 0.3 * hydro_score

        # Estimate KD (very rough approximation)
        # In reality, this would come from docking
        estimated_kd = self._score_to_kd(binding_score)

        return {
            "binding_score": binding_score,
            "estimated_kd": estimated_kd,
            "kd_unit": "nM",
            "confidence": "low",  # Heuristic only
        }

    def _compute_hydrophobic_moment(self, seq: str) -> float:
        """Compute simplified hydrophobic moment."""
        if len(seq) < 3:
            return 0.5

        # Check for alternating hydrophobic pattern
        hydro_values = [AA_PROPERTIES.get(aa, 0) for aa in seq]
        changes = sum(
            1 for i in range(len(hydro_values) - 1)
            if (hydro_values[i] > 0) != (hydro_values[i+1] > 0)
        )

        return min(1.0, changes / (len(seq) - 1) * 2)

    def _score_to_kd(self, score: float) -> float:
        """Convert binding score to estimated KD."""
        # Very rough approximation
        # Higher score = lower KD (tighter binding)
        if score >= 0.8:
            return np.random.uniform(0.1, 1.0)  # nM range
        elif score >= 0.6:
            return np.random.uniform(1.0, 10.0)  # nM range
        elif score >= 0.4:
            return np.random.uniform(10.0, 100.0)  # nM range
        else:
            return np.random.uniform(100.0, 1000.0)  # nM range


# ==================== CLI ====================
def main():
    """CLI entry point for candidate scoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Score CDR3 candidates")
    parser.add_argument("--input", type=str, required=True, help="Input candidates JSON")
    parser.add_argument("--output", type=str, default="ranked.json", help="Output file")
    parser.add_argument("--top-n", type=int, default=10, help="Top N candidates")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load candidates
    with open(args.input, "r") as f:
        candidates = json.load(f)

    # Score
    scorer = CandidateScorer()
    ranked = scorer.rank_candidates(candidates, top_n=args.top_n)

    # Save
    with open(args.output, "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"Ranked {len(ranked)} candidates, saved to {args.output}")

    # Print top 5
    print("\nTop 5 candidates:")
    for i, c in enumerate(ranked[:5]):
        print(f"{i+1}. {c['sequence']} (score: {c['final_score']:.3f})")


if __name__ == "__main__":
    main()
