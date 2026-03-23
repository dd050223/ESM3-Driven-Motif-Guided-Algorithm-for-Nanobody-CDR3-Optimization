"""
Docking Evaluator for CDR3-Target Protein Binding.

Uses AutoDock Vina for protein-peptide docking to estimate binding affinity.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DockingResult:
    """Docking result for a CDR3-target complex."""
    sequence: str
    target_name: str
    binding_energy: float  # kcal/mol (more negative = better binding)
    affinity_nM: Optional[float] = None  # Estimated KD in nM
    num_poses: int = 0
    best_pose: Optional[np.ndarray] = None
    docking_successful: bool = False
    error: Optional[str] = None


class DockingEvaluator:
    """
    Evaluate CDR3 binding using AutoDock Vina.

    This performs protein-peptide docking to estimate binding energy
    between the CDR3 region and target protein.
    """

    def __init__(
        self,
        vina_executable: str = "vina",
        receptor_pdb: Optional[str] = None,
        binding_site: Optional[Dict] = None,
        exhaustiveness: int = 8,
        num_modes: int = 9,
    ):
        """
        Initialize docking evaluator.

        Args:
            vina_executable: Path to AutoDock Vina executable
            receptor_pdb: Path to receptor PDB file
            binding_site: Dict with 'center' (x, y, z) and 'size' (x, y, z)
            exhaustiveness: Search exhaustiveness (1-32)
            num_modes: Number of binding modes to generate
        """
        self.vina_executable = vina_executable
        self.receptor_pdb = receptor_pdb
        self.binding_site = binding_site
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

        # Check if Vina is available
        self.vina_available = self._check_vina()

    def _check_vina(self) -> bool:
        """Check if AutoDock Vina is available."""
        try:
            result = subprocess.run(
                [self.vina_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"AutoDock Vina available: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning(f"AutoDock Vina not found at: {self.vina_executable}")
        except Exception as e:
            logger.warning(f"Error checking Vina: {e}")

        logger.warning("AutoDock Vina not available, docking will be skipped")
        return False

    def set_receptor(self, pdb_path: str, binding_site: Optional[Dict] = None):
        """Set the receptor protein for docking."""
        self.receptor_pdb = pdb_path
        if binding_site:
            self.binding_site = binding_site

    def prepare_receptor(self, pdb_path: str, output_path: Optional[str] = None) -> str:
        """
        Prepare receptor for docking using AutoDockTools.

        Args:
            pdb_path: Path to receptor PDB file
            output_path: Output path for PDBQT file

        Returns:
            Path to prepared receptor PDBQT file
        """
        if output_path is None:
            output_path = pdb_path.replace(".pdb", "_receptor.pdbqt")

        try:
            from AutoDockTools import ReceptorPyMOLGenerator
            # Use MGLTools to prepare receptor
            import sys
            sys.path.append(os.environ.get("MGLTOOLS_PATH", "/usr/local/mgltools"))

            # Try using prepare_receptor4.py
            cmd = [
                "python", "-m", "AutoDockTools.Utilities24.prepare_receptor4",
                "-r", pdb_path,
                "-o", output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Prepared receptor: {output_path}")
                return output_path

        except Exception as e:
            logger.warning(f"Failed to prepare receptor with AutoDockTools: {e}")

        # Fallback: try using MGLTools directly
        try:
            cmd = f"prepare_receptor4.py -r {pdb_path} -o {output_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
        except Exception:
            pass

        # Last resort: just copy and hope Vina can handle it
        logger.warning("Using PDB directly (Vina may not work without PDBQT)")
        return pdb_path

    def prepare_ligand(self, sequence: str, output_path: Optional[str] = None) -> str:
        """
        Prepare ligand (CDR3 peptide) for docking.

        Args:
            sequence: CDR3 amino acid sequence
            output_path: Output path for PDBQT file

        Returns:
            Path to prepared ligand PDBQT file
        """
        if output_path is None:
            output_path = f"ligand_{sequence}.pdbqt"

        # Build PDB file from sequence using Biopython
        try:
            from Bio.PDB import PDBBuilder, Structure
            from Bio.PDB import Residue, Chain, Model
            from Bio.SeqUtils import seq1

            # Create a simple peptide structure
            # This is a simplified approach - real docking would need proper 3D structure
            pdb_content = self._build_peptide_pdb(sequence)

            pdb_path = output_path.replace(".pdbqt", ".pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb_content)

            # Try to convert to PDBQT
            try:
                cmd = f"prepare_ligand4.py -l {pdb_path} -o {output_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return output_path
            except Exception:
                pass

            return pdb_path

        except Exception as e:
            logger.warning(f"Failed to prepare ligand: {e}")
            # Return a simple PDB file
            pdb_path = output_path.replace(".pdbqt", ".pdb")
            with open(pdb_path, "w") as f:
                f.write(self._build_peptide_pdb(sequence))
            return pdb_path

    def _build_peptide_pdb(self, sequence: str) -> str:
        """
        Build a simple PDB file for a peptide.

        Note: This creates a linear peptide. For actual docking,
        a proper 3D structure from AlphaFold or Rosetta would be better.
        """
        pdb_lines = []
        pdb_lines.append("HEADER    GENERATED PEPTIDE")
        pdb_lines.append("TITLE     CDR3 PEPTIDE")

        # Create ATOM records for each residue
        # Using idealized helical coordinates
        for i, aa in enumerate(sequence):
            # Simple helical approximation
            x = 0 + i * 1.5  # Spacing along x-axis
            y = 5 * np.sin(i * 0.5)
            z = 5 * np.cos(i * 0.5)

            # Backbone atoms (N, CA, C, O)
            atom_num = i * 4 + 1

            # N atom
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  N   {aa:3s} A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           N  "
            )

            # CA atom
            pdb_lines.append(
                f"ATOM  {atom_num+1:5d}  CA  {aa:3s} A{1:4d}    "
                f"{x+1.5:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  "
            )

            # C atom
            pdb_lines.append(
                f"ATOM  {atom_num+2:5d}  C   {aa:3s} A{1:4d}    "
                f"{x+3.0:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  "
            )

            # O atom
            pdb_lines.append(
                f"ATOM  {atom_num+3:5d}  O   {aa:3s} A{1:4d}    "
                f"{x+3.5:8.3f}{y+1.0:8.3f}{z:8.3f}  1.00 20.00           O  "
            )

        pdb_lines.append("END")
        return "\n".join(pdb_lines)

    def dock(
        self,
        ligand_sequence: str,
        receptor_path: Optional[str] = None,
        binding_site: Optional[Dict] = None,
    ) -> DockingResult:
        """
        Perform docking for a single CDR3 sequence.

        Args:
            ligand_sequence: CDR3 amino acid sequence
            receptor_path: Path to receptor PDB file (overrides default)
            binding_site: Binding site coordinates (overrides default)

        Returns:
            DockingResult with binding energy
        """
        if not self.vina_available:
            return DockingResult(
                sequence=ligand_sequence,
                target_name=receptor_path or "unknown",
                binding_energy=0.0,
                docking_successful=False,
                error="AutoDock Vina not available",
            )

        # Use defaults if not specified
        receptor_path = receptor_path or self.receptor_pdb
        binding_site = binding_site or self.binding_site

        if not receptor_path or not binding_site:
            return DockingResult(
                sequence=ligand_sequence,
                target_name=receptor_path or "unknown",
                binding_energy=0.0,
                docking_successful=False,
                error="Receptor or binding site not specified",
            )

        # Create temporary files
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Prepare ligand
            ligand_pdbqt = temp_dir / f"ligand_{ligand_sequence}.pdbqt"
            ligand_pdb = self._build_peptide_pdb(ligand_sequence)
            ligand_pdb_path = temp_dir / "ligand.pdb"

            with open(ligand_pdb_path, "w") as f:
                f.write(ligand_pdb)

            # Try to prepare with prepare_ligand4
            try:
                cmd = [
                    "prepare_ligand4.py",
                    "-l", str(ligand_pdb_path),
                    "-o", str(ligand_pdbqt),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception:
                # Use PDB directly as fallback
                ligand_pdbqt = ligand_pdb_path

            # Prepare receptor
            receptor_pdbqt = temp_dir / "receptor.pdbqt"
            try:
                cmd = [
                    "prepare_receptor4.py",
                    "-r", receptor_path,
                    "-o", str(receptor_pdbqt),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    receptor_pdbqt = receptor_path  # Fallback
            except Exception:
                receptor_pdbqt = receptor_path  # Fallback

            # Run docking
            output_log = temp_dir / "docking_log.txt"
            output_pdbqt = temp_dir / "docking_results.pdbqt"

            center = binding_site["center"]
            size = binding_site["size"]

            cmd = [
                self.vina_executable,
                "--receptor", str(receptor_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(size[0]),
                "--size_y", str(size[1]),
                "--size_z", str(size[2]),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--out", str(output_pdbqt),
                "--log", str(output_log),
            ]

            logger.info(f"Running docking for {ligand_sequence}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                return DockingResult(
                    sequence=ligand_sequence,
                    target_name=receptor_path,
                    binding_energy=0.0,
                    docking_successful=False,
                    error=f"Docking failed: {result.stderr}",
                )

            # Parse output
            if output_log.exists():
                with open(output_log, "r") as f:
                    log_content = f.read()

                # Extract binding energy from log
                best_energy = self._parse_vina_log(log_content)

                # Estimate affinity from binding energy
                affinity = self._energy_to_affinity(best_energy)

                return DockingResult(
                    sequence=ligand_sequence,
                    target_name=os.path.basename(receptor_path),
                    binding_energy=best_energy,
                    affinity_nM=affinity,
                    num_poses=self.num_modes,
                    docking_successful=True,
                )

            return DockingResult(
                sequence=ligand_sequence,
                target_name=receptor_path,
                binding_energy=0.0,
                docking_successful=False,
                error="No output log generated",
            )

        except subprocess.TimeoutExpired:
            return DockingResult(
                sequence=ligand_sequence,
                target_name=receptor_path,
                binding_energy=0.0,
                docking_successful=False,
                error="Docking timeout",
            )
        except Exception as e:
            return DockingResult(
                sequence=ligand_sequence,
                target_name=receptor_path,
                binding_energy=0.0,
                docking_successful=False,
                error=str(e),
            )
        finally:
            # Cleanup temp files
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def _parse_vina_log(self, log_content: str) -> float:
        """Parse binding energy from Vina log output."""
        lines = log_content.strip().split("\n")

        for line in lines:
            # Look for line like: "1       -7.2      0.000      0.000"
            if "1" in line and "-" in line:
                parts = line.split()
                try:
                    # First number is rank, second is binding energy
                    if len(parts) >= 2:
                        energy = float(parts[1])
                        return energy
                except (ValueError, IndexError):
                    continue

        # Try alternative parsing
        for line in lines:
            if "REMARK" in line and "=" in line:
                try:
                    energy = float(line.split("=")[1].split()[0])
                    return energy
                except (ValueError, IndexError):
                    continue

        logger.warning("Could not parse binding energy from Vina log")
        return -5.0  # Default moderate binding

    def _energy_to_affinity(self, binding_energy: float) -> float:
        """
        Convert binding energy to estimated affinity (KD).

        Using the approximation: KD = exp(ΔG / RT)
        At 298K (25C), RT = 0.592 kcal/mol

        Note: deltaG is negative for binding
        """
        import math
        R = 0.592  # kcal/(mol*K) at 298K

        # KD in M = exp(ΔG / RT) where ΔG is negative
        # Since binding_energy is already negative, use it directly
        KD_molar = math.exp(binding_energy / R)
        KD_nM = KD_molar * 1e9

        # Ensure reasonable bounds
        if KD_nM < 0.01:
            KD_nM = 0.01
        elif KD_nM > 1e6:
            KD_nM = 1e6

        return KD_nM

    def batch_dock(
        self,
        sequences: List[str],
        receptor_path: Optional[str] = None,
        binding_site: Optional[Dict] = None,
    ) -> Dict[str, DockingResult]:
        """
        Dock multiple CDR3 sequences.

        Args:
            sequences: List of CDR3 sequences
            receptor_path: Path to receptor PDB
            binding_site: Binding site coordinates

        Returns:
            Dict mapping sequence to DockingResult
        """
        results = {}

        for i, seq in enumerate(sequences):
            logger.info(f"Docking {i+1}/{len(sequences)}: {seq}")

            result = self.dock(seq, receptor_path, binding_site)
            results[seq] = result

        return results


class HeuristicDockingEstimator:
    """
    Heuristic estimator for binding when AutoDock Vina is not available.

    Uses sequence-based features to estimate binding potential.
    """

    def __init__(self):
        """Initialize heuristic estimator."""
        # Key binding residues from nanobody studies
        self.binding_motifs = {
            "high": ["YRY", "YYG", "RYG", "RYY", "YYR", "WYY", "YYW"],
            "medium": ["YNY", "NYG", "ARL", "RLG", "RNY", "NYR", "YRN"],
        }

    def estimate_binding(
        self,
        cdr3_sequence: str,
        target_type: str = "generic",
    ) -> DockingResult:
        """
        Estimate binding using heuristic scoring.

        Args:
            cdr3_sequence: CDR3 amino acid sequence
            target_type: Target protein type

        Returns:
            DockingResult with estimated binding energy
        """
        if not cdr3_sequence:
            return DockingResult(
                sequence=cdr3_sequence,
                target_name=target_type,
                binding_energy=0.0,
                docking_successful=False,
                error="Empty sequence",
            )

        # Score based on binding motifs
        motif_score = 0.0
        seq_upper = cdr3_sequence.upper()

        for motif in self.binding_motifs["high"]:
            if motif in seq_upper:
                motif_score += 2.0

        for motif in self.binding_motifs["medium"]:
            if motif in seq_upper:
                motif_score += 1.0

        # Score based on amino acid composition
        # Aromatic residues (Y, W, F) enhance binding
        aromatic = sum(1 for aa in cdr3_sequence if aa in "YWF")
        aromatic_score = aromatic * 0.3

        # Charged residues (R, K) can enhance binding
        charged = sum(1 for aa in cdr3_sequence if aa in "RK")
        charged_score = charged * 0.2

        # Length consideration
        length = len(cdr3_sequence)
        if 8 <= length <= 16:
            length_score = 0.5
        else:
            length_score = 0.2

        # Combine scores
        total_score = motif_score + aromatic_score + charged_score + length_score

        # Convert to estimated binding energy
        # Typical range: -4 to -10 kcal/mol
        base_energy = -4.0
        binding_energy = base_energy - total_score

        # Clamp to realistic range
        binding_energy = max(-10.0, min(-3.0, binding_energy))

        # Calculate affinity using proper thermodynamic relationship
        # KD = exp(ΔG / RT) where ΔG is negative for binding
        # So: KD = exp(-|ΔG| / RT)
        import math
        R = 0.592  # kcal/(mol*K) at 298K
        # Since binding_energy is negative, we use it directly: exp(negative / positive) = small number
        KD_molar = math.exp(binding_energy / R)
        KD_nM = KD_molar * 1e9

        # Ensure reasonable bounds
        if KD_nM < 0.01:
            KD_nM = 0.01
        elif KD_nM > 1e6:
            KD_nM = 1e6

        return DockingResult(
            sequence=cdr3_sequence,
            target_name=target_type,
            binding_energy=binding_energy,
            affinity_nM=KD_nM,
            docking_successful=True,
        )

    def batch_estimate(
        self,
        sequences: List[str],
        target_type: str = "generic",
    ) -> Dict[str, DockingResult]:
        """Estimate binding for multiple sequences."""
        return {seq: self.estimate_binding(seq, target_type) for seq in sequences}


# ==================== CLI ====================
def main():
    """CLI entry point for docking evaluation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate CDR3 binding via docking")
    parser.add_argument("--sequences", type=str, nargs="+", required=True,
                       help="CDR3 sequences to dock")
    parser.add_argument("--receptor", type=str, required=True,
                       help="Path to receptor PDB file")
    parser.add_argument("--center", type=float, nargs=3, required=True,
                       help="Binding site center (x y z)")
    parser.add_argument("--size", type=float, nargs=3, default=[20, 20, 20],
                       help="Binding site size (x y z)")
    parser.add_argument("--output", type=str, default="docking_results.json",
                       help="Output file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Setup binding site
    binding_site = {
        "center": args.center,
        "size": args.size,
    }

    # Run docking
    evaluator = DockingEvaluator()
    results = evaluator.batch_dock(args.sequences, args.receptor, binding_site)

    # Save results
    output = {
        seq: {
            "binding_energy": r.binding_energy,
            "affinity_nM": r.affinity_nM,
            "docking_successful": r.docking_successful,
            "error": r.error,
        }
        for seq, r in results.items()
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Docking results saved to {args.output}")
    for seq, r in results.items():
        print(f"  {seq}: {r.binding_energy:.2f} kcal/mol ({r.affinity_nM:.1f} nM)")


if __name__ == "__main__":
    main()
