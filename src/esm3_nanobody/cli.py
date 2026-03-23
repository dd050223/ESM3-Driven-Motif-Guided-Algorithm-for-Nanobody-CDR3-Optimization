"""
Command-line interface for ESM3 Nanobody CDR3 Optimization.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import argparse

from .generator import CDR3Generator, GenerationConfig
from .structure_predictor import StructurePredictor
from .scorer import CandidateScorer, ScorerConfig
from .docking_evaluator import DockingEvaluator, HeuristicDockingEstimator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_candidates(candidates: list, output_path: str):
    """Save candidates to JSON file."""
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)


def run_pipeline(config: dict) -> dict:
    """
    Run the complete CDR3 optimization pipeline.

    Steps:
    1. Generate CDR3 candidates with ESM3
    2. Predict structures
    3. Score and rank candidates
    4. Save results

    Args:
        config: Configuration dictionary

    Returns:
        Summary dictionary with results
    """
    output_dir = Path(config.get("output_dir", "outputs/default_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting ESM3 Nanobody CDR3 Optimization Pipeline")
    logger.info("=" * 60)

    # Extract configuration
    model_path = config.get("model_path", "E:/teacherstudnet model/esm3_sm_open_v1.pth")
    framework_seq = config.get("framework_sequence", config.get("target_sequence"))

    if not framework_seq:
        logger.error("No framework sequence provided in config!")
        return {"error": "No framework sequence provided"}

    device = config.get("device", "auto")
    backend = config.get("backend", "esm3")

    # Generation config
    gen_config = GenerationConfig(
        temperature=config.get("temperature", 0.8),
        top_k=config.get("top_k", 40),
        top_p=config.get("top_p", 0.9),
        num_candidates=config.get("num_candidates", 32),
        refinement_rounds=config.get("refinement_rounds", 3),
        remask_fraction=config.get("remask_fraction", 0.3),
        min_length=config.get("cdr3_length_min", 8),
        max_length=config.get("cdr3_length_max", 20),
    )

    # Scorer config
    weights = config.get("weights", {})
    scorer_config = ScorerConfig(
        weight_affinity=weights.get("affinity", 0.3),
        weight_plddt=weights.get("plddt", 0.25),
        weight_diversity=weights.get("diversity", 0.15),
        weight_stability=weights.get("stability", 0.15),
        weight_docking=weights.get("docking", 0.15),
    )

    logger.info(f"Framework sequence: {framework_seq[:50]}...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Generating {gen_config.num_candidates} candidates...")

    # Step 1: Generate CDR3 candidates
    logger.info("\n[Step 1/4] Generating CDR3 candidates with ESM3...")

    try:
        generator = CDR3Generator(model_path, gen_config, device)
        candidates = generator.generate_candidates(framework_seq)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Fallback to mock generation for testing
        logger.info("Falling back to mock generation...")
        gen_config_ = GenerationConfig(num_candidates=config.get("num_candidates", 32))
        generator = CDR3Generator("", gen_config_, device)
        candidates = generator.generate_candidates(framework_seq)

    logger.info(f"Generated {len(candidates)} candidates")

    # Save raw candidates
    save_candidates(candidates, output_dir / "raw_candidates.json")

    # Step 2: Structure prediction (use same ESM3 model)
    logger.info("\n[Step 2/4] Predicting structures...")

    # Pass the loaded ESM3 model to structure predictor for efficiency
    predictor = StructurePredictor(
        predictor_type="esm3",
        device=device,
        esm3_model=generator.esm3.model if generator.esm3.model_loaded else None,
        model_path=model_path,
    )

    structure_results = {}
    for i, candidate in enumerate(candidates):
        seq = candidate.get("sequence", "")
        if not seq:
            continue

        if i % 10 == 0:
            logger.info(f"Predicting structure {i+1}/{len(candidates)}")

        try:
            result = predictor.predict(seq)
            structure_results[seq] = {
                "mean_plddt": result.mean_plddt,
                "cdr3_plddt": result.cdr3_plddt,
            }
        except Exception as e:
            logger.warning(f"Structure prediction failed for {seq}: {e}")
            structure_results[seq] = {
                "mean_plddt": 70.0,  # Default
                "cdr3_plddt": 65.0,
            }

    # Step 3: Docking evaluation (optional)
    logger.info("\n[Step 3/4] Evaluating binding affinity...")

    docking_results = {}
    use_docking = config.get("use_docking", False)

    if use_docking:
        # Check for target protein config
        target_pdb = config.get("target_pdb", None)
        binding_site = config.get("binding_site", None)

        if target_pdb and binding_site:
            # Try real docking with AutoDock Vina
            docking_eval = DockingEvaluator(
                receptor_pdb=target_pdb,
                binding_site=binding_site,
            )

            if docking_eval.vina_available:
                sequences = [c.get("sequence", "") for c in candidates if c.get("sequence")]
                docking_results_raw = docking_eval.batch_dock(sequences, target_pdb, binding_site)

                for seq, result in docking_results_raw.items():
                    docking_results[seq] = {
                        "binding_energy": result.binding_energy,
                        "affinity_nM": result.affinity_nM,
                        "docking_successful": result.docking_successful,
                    }
                logger.info(f"Docking completed for {len(docking_results)} sequences")
            else:
                logger.warning("AutoDock Vina not available, using heuristic estimation")
                use_docking = False

        else:
            logger.info("No target protein specified, using heuristic binding estimation")
            use_docking = False

    if not use_docking or not docking_results:
        # Use heuristic binding estimator
        heuristic_estimator = HeuristicDockingEstimator()
        sequences = [c.get("sequence", "") for c in candidates if c.get("sequence")]

        for seq in sequences:
            result = heuristic_estimator.estimate_binding(seq, config.get("target_type", "generic"))
            docking_results[seq] = {
                "binding_energy": result.binding_energy,
                "affinity_nM": result.affinity_nM,
                "docking_successful": True,
            }

        logger.info(f"Heuristic binding estimation completed for {len(docking_results)} sequences")

    # Step 4: Score and rank
    logger.info("\n[Step 4/4] Scoring and ranking candidates...")

    scorer = CandidateScorer(scorer_config)
    ranked_candidates = scorer.rank_candidates(
        candidates,
        structure_results=structure_results,
        docking_results=docking_results,
        top_n=config.get("save_top_n", 10),
    )

    # Save ranked candidates
    save_candidates(ranked_candidates, output_dir / "candidates.csv")

    # Also save as JSON for full data
    with open(output_dir / "candidates.json", "w") as f:
        json.dump(ranked_candidates, f, indent=2)

    # Save summary
    summary = {
        "total_candidates": len(candidates),
        "unique_candidates": len(set(c.get("sequence", "") for c in candidates)),
        "top_candidates": [
            {
                "rank": c.get("rank"),
                "sequence": c.get("sequence"),
                "final_score": c.get("final_score"),
                "plddt_score": c.get("plddt_score"),
                "affinity_score": c.get("affinity_score"),
                "docking_score": c.get("docking_score"),
                "binding_energy": c.get("binding_energy"),
                "estimated_kd": c.get("estimated_kd"),
            }
            for c in ranked_candidates[:10]
        ],
        "config": {
            "temperature": gen_config.temperature,
            "top_k": gen_config.top_k,
            "num_candidates": gen_config.num_candidates,
            "refinement_rounds": gen_config.refinement_rounds,
            "use_docking": use_docking,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save top candidates as text
    with open(output_dir / "top_candidates.txt", "w") as f:
        f.write("Top CDR3 Candidates\n")
        f.write("=" * 40 + "\n\n")
        for c in ranked_candidates[:10]:
            f.write(f"Rank {c.get('rank')}: {c.get('sequence')}\n")
            f.write(f"  Score: {c.get('final_score'):.4f}\n")
            f.write(f"  pLDDT Score: {c.get('plddt_score'):.4f}\n")
            f.write(f"  Affinity Score: {c.get('affinity_score'):.4f}\n")
            f.write(f"  Docking Score: {c.get('docking_score'):.4f}\n")
            f.write(f"  Binding Energy: {c.get('binding_energy', 0):.2f} kcal/mol\n")
            if c.get('estimated_kd'):
                f.write(f"  Estimated KD: {c.get('estimated_kd'):.1f} nM\n")
            f.write(f"  Diversity Score: {c.get('diversity_score'):.4f}\n")
            f.write(f"  Stability Score: {c.get('stability_score'):.4f}\n\n")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)

    # Print top candidates
    print("\nTop 10 CDR3 Candidates:")
    print("-" * 50)
    for c in ranked_candidates[:10]:
        print(f"{c.get('rank'):2d}. {c.get('sequence'):20s} | Score: {c.get('final_score'):.4f}")

    return summary


def download_data(output_dir: str) -> dict:
    """
    Download example data from RCSB PDB and UniProt.

    Args:
        output_dir: Directory to save downloaded data

    Returns:
        Summary of downloaded files
    """
    from .data_utils import download_pdb, download_uniprot

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # PDB IDs to download
    pdb_ids = ["1MWE", "2P42", "5E5E"]  # VHH structures

    # UniProt IDs
    uniprot_ids = {
        "P0DTC2": "spike",
        "P04626": "HER2",
    }

    downloaded = {}

    # Download PDB files
    for pdb_id in pdb_ids:
        try:
            logger.info(f"Downloading PDB {pdb_id}...")
            filepath = download_pdb(pdb_id, output_path)
            downloaded[pdb_id] = str(filepath)
            logger.info(f"  Saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to download {pdb_id}: {e}")

    # Download UniProt sequences
    for uniprot_id, name in uniprot_ids.items():
        try:
            logger.info(f"Downloading UniProt {uniprot_id} ({name})...")
            filepath = download_uniprot(uniprot_id, output_path)
            downloaded[uniprot_id] = str(filepath)
            logger.info(f"  Saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to download {uniprot_id}: {e}")

    return downloaded




def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ESM3 Nanobody CDR3 Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the optimization pipeline")
    run_parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Download command
    download_parser = subparsers.add_parser("download-data", help="Download example data")
    download_parser.add_argument("--output-dir", type=str, default="data/downloaded", help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    # Execute command
    if args.command == "run":
        config = load_config(args.config)
        if args.verbose:
            config["verbose"] = True
        run_pipeline(config)

    elif args.command == "download-data":
        download_data(args.output_dir)


if __name__ == "__main__":
    main()
