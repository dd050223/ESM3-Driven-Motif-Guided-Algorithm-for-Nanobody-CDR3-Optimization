"""
Data download utilities for PDB and UniProt.
"""

import os
import requests
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """
    Download PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB ID
        output_dir: Directory to save file

    Returns:
        Path to downloaded file
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = output_dir / f"{pdb_id}.pdb"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(output_path, "w") as f:
        f.write(response.text)

    return output_path


def download_uniprot(uniprot_id: str, output_dir: Path) -> Path:
    """
    Download UniProt sequence in FASTA format.

    Args:
        uniprot_id: UniProt accession ID
        output_dir: Directory to save file

    Returns:
        Path to downloaded file
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    output_path = output_dir / f"{uniprot_id}.fasta"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(output_path, "w") as f:
        f.write(response.text)

    return output_path


def parse_pdb_sequence(pdb_path: str) -> str:
    """
    Extract protein sequence from PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Protein sequence
    """
    sequence = []
    seen_residues = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract residue info
                try:
                    resname = line[17:20].strip()
                    chain = line[21]
                    resnum = int(line[22:26])
                    atomname = line[12:16].strip()

                    # Only take CA atoms for sequence
                    if atomname == "CA" and resname not in ["HOH", "MSE"]:
                        key = (chain, resnum)
                        if key not in seen_residues:
                            # Convert 3-letter to 1-letter code
                            aa = three_to_one(resname)
                            if aa:
                                sequence.append(aa)
                                seen_residues.add(key)

                except (ValueError, IndexError):
                    continue

    return "".join(sequence)


def three_to_one(aa: str) -> Optional[str]:
    """Convert 3-letter amino acid code to 1-letter."""
    mapping = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "MSE": "M",  # Selenomethionine
    }
    return mapping.get(aa.upper())


def extract_cdr_from_pdb(
    pdb_path: str,
    chain: str = "A",
    numbering: str = "kabat",
) -> dict:
    """
    Extract CDR regions from PDB file.

    Args:
        pdb_path: Path to PDB file
        chain: Chain ID
        numbering: Numbering scheme ("kabat" or "imgt")

    Returns:
        Dictionary with framework and CDR regions
    """
    # CDR definitions (Kabat numbering)
    cdr_regions_kabat = {
        "CDR1": (26, 38),
        "CDR2": (56, 65),
        "CDR3": (95, 110),
    }

    # CDR definitions (IMGT numbering)
    cdr_regions_imgt = {
        "CDR1": (27, 38),
        "CDR2": (56, 65),
        "CDR3": (105, 117),
    }

    cdr_regions = cdr_regions_kabat if numbering == "kabat" else cdr_regions_imgt

    # Parse PDB
    residues = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    if line[21] != chain:
                        continue

                    resname = line[17:20].strip()
                    resnum = int(line[22:26])
                    atomname = line[12:16].strip()

                    if atomname == "CA" and resname not in ["HOH", "MSE"]:
                        aa = three_to_one(resname)
                        if aa:
                            residues[resnum] = aa

                except (ValueError, IndexError):
                    continue

    # Extract regions
    result = {"full_sequence": ""}

    # Build full sequence from residues
    for resnum in sorted(residues.keys()):
        result["full_sequence"] += residues[resnum]

    # Extract CDRs
    for cdr_name, (start, end) in cdr_regions.items():
        cdr_seq = ""
        for resnum in range(start, end + 1):
            if resnum in residues:
                cdr_seq += residues[resnum]
        result[cdr_name] = cdr_seq

    # Framework regions
    result["FR1"] = result["full_sequence"][:25]
    result["FR2"] = result["full_sequence"][38:55]
    result["FR3"] = result["full_sequence"][65:94]
    result["FR4"] = result["full_sequence"][110:]

    return result


def parse_fasta(fasta_path: str) -> dict:
    """
    Parse FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dictionary with sequence info
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences.append({
                        "id": current_id,
                        "sequence": "".join(current_seq),
                    })
                # Parse header
                parts = line[1:].split()
                current_id = parts[0] if parts else "unknown"
                current_seq = []
            else:
                current_seq.append(line)

        # Add last sequence
        if current_id:
            sequences.append({
                "id": current_id,
                "sequence": "".join(current_seq),
            })

    return sequences[0] if sequences else {"id": None, "sequence": ""}


# Example usage
if __name__ == "__main__":
    # Test downloading
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download 1MWE (VHH structure)
        pdb_path = download_pdb("1MWE", Path(tmpdir))
        print(f"Downloaded: {pdb_path}")

        # Parse sequence
        seq = parse_pdb_sequence(str(pdb_path))
        print(f"Sequence: {seq[:50]}...")

        # Extract CDRs
        cdrs = extract_cdr_from_pdb(str(pdb_path))
        print(f"CDR1: {cdrs.get('CDR1')}")
        print(f"CDR2: {cdrs.get('CDR2')}")
        print(f"CDR3: {cdrs.get('CDR3')}")
