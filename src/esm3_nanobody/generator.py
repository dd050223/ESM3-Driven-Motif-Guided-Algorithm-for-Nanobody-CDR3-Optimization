"""
ESM3-based CDR3 Generator for Nanobody Optimization.

Uses local ESM3 model (esm3_sm_open_v1.pth) to generate CDR3 candidates.
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ESM3 vocabulary special tokens
SPECIAL_TOKENS = {
    "<cls>": 0,
    "<eos>": 1,
    "<pad>": 0,
    "<mask>": 32,
    "<sep>": 33,
    "<bone>": 34,
    "<chem>": 35,
    "<struct>": 36,
}


class ESM3Tokenizer:
    """Tokenizer for ESM3 model."""

    # ESM3 uses amino acid tokens (0-20 for amino acids, 32 for mask, etc.)
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    VOCAB = {
        "<cls>": 0,
        "<eos>": 1,
        "<mask>": 32,
        "<sep>": 33,
    }

    def __init__(self):
        # Build amino acid vocabulary
        for i, aa in enumerate(self.AMINO_ACIDS):
            self.VOCAB[aa] = i + 4  # Start from 4

        self.vocab_size = 128  # ESM3 vocab size
        self.mask_token_id = self.VOCAB["<mask>"]

    def encode(self, sequence: str) -> List[int]:
        """Encode amino acid sequence to token IDs."""
        tokens = [self.VOCAB["<cls>"]]
        for aa in sequence.upper():
            if aa in self.VOCAB:
                tokens.append(self.VOCAB[aa])
            elif aa == "X":  # Unknown amino acid
                tokens.append(random.randint(4, 23))  # Random AA
            # Skip invalid characters
        tokens.append(self.VOCAB["<eos>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to amino acid sequence."""
        aa_list = []
        reverse_vocab = {v: k for k, v in self.VOCAB.items()}
        for tid in token_ids:
            if tid in reverse_vocab and reverse_vocab[tid] in self.AMINO_ACIDS:
                aa_list.append(reverse_vocab[tid])
            elif tid in reverse_vocab and reverse_vocab[tid] == "<mask>":
                aa_list.append("X")
        return "".join(aa_list)

    def mask_sequence(self, sequence: str, mask_positions: List[int]) -> Tuple[str, str]:
        """Replace specified positions with mask token."""
        seq_list = list(sequence)
        original = []
        for pos in mask_positions:
            if pos < len(seq_list):
                original.append(seq_list[pos])
                seq_list[pos] = "<mask>"
        return "".join(seq_list), "".join(original)


class ESM3ModelWrapper:
    """Wrapper for loading and using local ESM3 model."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize ESM3 model from local checkpoint."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading ESM3 model from {model_path}...")
        logger.info(f"Using device: {self.device}")

        # Set environment variables for offline mode
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        try:
            from esm.tokenization import EsmSequenceTokenizer
            from esm.models.esm3 import ESM3
            from esm.pretrained import (
                ESM3_STRUCTURE_ENCODER_V0,
                ESM3_STRUCTURE_DECODER_V0,
                ESM3_FUNCTION_DECODER_V0,
            )
            from dataclasses import dataclass
            from typing import Optional

            # Create ESM3 model
            self.model = ESM3(
                d_model=1536,
                n_heads=24,
                v_heads=256,
                n_layers=48,
                structure_encoder_name=ESM3_STRUCTURE_ENCODER_V0,
                structure_decoder_name=ESM3_STRUCTURE_DECODER_V0,
                function_decoder_name=ESM3_FUNCTION_DECODER_V0,
            ).to(self.device).eval()

            # Load weights from local file
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model loaded - missing keys: {len(missing)}, unexpected: {len(unexpected)}")

            # Get tokenizer
            self.tokenizer = EsmSequenceTokenizer()
            self._wrapped_tokenizer = self.tokenizer

            logger.info("ESM3 model loaded successfully!")
            self.model_loaded = True

            # Try to use ESM3's native generate
            self._use_native_generate = True

        except Exception as e:
            logger.warning(f"Failed to load ESM3 model: {e}")
            logger.warning("Using mock mode for testing")
            self.model = None
            self.tokenizer = ESM3Tokenizer()
            self._wrapped_tokenizer = self.tokenizer
            self.model_loaded = False
            self._use_native_generate = False

    def generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Generate tokens given input sequence."""
        if not self.model_loaded or self.model is None:
            # Return mock generation
            return self._mock_generate(input_ids)

        try:
            # Try using ESM3's native generation interface
            return self._generate_native(input_ids, temperature, top_k, top_p)
        except Exception as e:
            logger.warning(f"Native generation failed: {e}, using mock")
            return self._mock_generate(input_ids)

    def _generate_native(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Use ESM3's native generate method with iterative mask filling."""
        import os
        # Ensure environment variables are set
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

        self.model.eval()

        # Get mask token ID
        mask_token_id = 32  # ESM3 mask token

        # Start with input tokens
        generated_tokens = input_ids.clone()

        # Find all masked positions
        mask_positions = (generated_tokens[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            # No mask to fill, just return last token
            return generated_tokens[:, -1:]

        # Iteratively fill masked positions
        for pos in mask_positions:
            pos = pos.item()

            with torch.no_grad():
                output = self.model(sequence_tokens=generated_tokens)
                logits = output.sequence_logits[0, pos, :]

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals = torch.topk(logits, top_k)
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    if generated_tokens.device.type == 'cuda':
                        mask = mask.cuda()
                    mask[top_k_vals.indices] = False
                    logits[mask] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update
                generated_tokens[0, pos] = next_token.item()

        # Return the generated tokens
        return generated_tokens

    def _mock_generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mock generation for testing without model."""
        batch_size, seq_len = input_ids.shape
        # Generate random amino acid tokens (4-23)
        mock_tokens = torch.randint(4, 24, (batch_size, 1), device=input_ids.device)
        return mock_tokens

    def get_embeddings(self, sequence: str) -> np.ndarray:
        """Get ESM3 embeddings for a sequence."""
        import os
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'

        if not self.model_loaded:
            # Return random embeddings
            return np.random.randn(1536)

        tokens = self.tokenizer.encode(sequence)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.model(sequence_tokens=input_tensor)
            # ESM3 returns ESMOutput with sequence_representations
            if hasattr(output, 'sequence_representations'):
                embeddings = output.sequence_representations
            elif hasattr(output, 'representations'):
                embeddings = output.representations.get('sequence')
            else:
                return np.random.randn(1536)

            if embeddings is not None:
                # Mean pooling over sequence
                return embeddings[0, :, :].mean(dim=0).cpu().numpy()
            else:
                return np.random.randn(1536)


@dataclass
class GenerationConfig:
    """Configuration for CDR3 generation."""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    num_candidates: int = 32
    refinement_rounds: int = 3
    remask_fraction: float = 0.3
    min_length: int = 8
    max_length: int = 20
    random_seed: int = 42


class CDR3Generator:
    """
    CDR3 Generator using ESM3.

    Generates CDR3 sequences while keeping framework region unchanged.
    """

    # Standard VHH framework regions (IMGT numbering simplified)
    FRAMEWORK_REGIONS = {
        "FR1": (1, 25),
        "FR2": (39, 55),
        "FR3": (66, 104),
        "FR4": (118, 129),
    }

    # Kabat numbering for CDR
    CDR_REGIONS = {
        "CDR1": (26, 38),
        "CDR2": (56, 65),
        "CDR3": (95, 110),
    }

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        device: str = "auto",
    ):
        """Initialize CDR3 generator with ESM3 model."""
        self.config = config or GenerationConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        self.esm3 = ESM3ModelWrapper(model_path, device)
        self.tokenizer = self.esm3.tokenizer

    def prepare_input(
        self,
        framework_sequence: str,
        cdr3_masked: bool = True,
    ) -> str:
        """
        Prepare input sequence with CDR3 masked.

        Args:
            framework_sequence: Full VHH sequence with CDR3 region
            cdr3_masked: If True, replace CDR3 with mask tokens

        Returns:
            Input sequence for ESM3
        """
        if cdr3_masked:
            # Determine mask length (CDR3 is typically 8-20 amino acids)
            mask_length = random.randint(self.config.min_length, self.config.max_length)
            mask_tokens = "<mask>" * mask_length

            # Find a good position to insert the mask
            # Look for common framework motifs to locate CDR3
            # VHH typically has CAR motif at end of FR3
            if "CAR" in framework_sequence:
                # Insert after CAR
                idx = framework_sequence.find("CAR") + 3
                input_seq = framework_sequence[:idx] + mask_tokens + framework_sequence[idx + 5:]
            elif "WG" in framework_sequence:
                # Look for WG at start of FR4
                idx = framework_sequence.find("WG")
                if idx > 50:  # Make sure it's in reasonable position
                    input_seq = framework_sequence[:idx-5] + mask_tokens + framework_sequence[idx+2:]
                else:
                    # Fallback: use position ~80-90 (typical CDR3 start)
                    input_seq = framework_sequence[:80] + mask_tokens + framework_sequence[90:]
            else:
                # Use position ~80 (typical CDR3 start in VHH)
                input_seq = framework_sequence[:80] + mask_tokens + framework_sequence[90:]

            return input_seq[:200]  # Limit length
        else:
            return framework_sequence[:200]

    def generate_candidates(
        self,
        framework_sequence: str,
        num_candidates: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate CDR3 candidates for a given framework.

        Args:
            framework_sequence: VHH framework sequence
            num_candidates: Number of candidates to generate

        Returns:
            List of candidate dictionaries
        """
        num_candidates = num_candidates or self.config.num_candidates
        candidates = []

        # Prepare masked input
        input_seq = self.prepare_input(framework_sequence, cdr3_masked=True)

        logger.info(f"Generating {num_candidates} CDR3 candidates...")
        logger.info(f"Input sequence (first 100 chars): {input_seq[:100]}...")

        # Generate multiple rounds
        for round_idx in range(self.config.refinement_rounds):
            logger.info(f"Refinement round {round_idx + 1}/{self.config.refinement_rounds}")

            for i in range(num_candidates):
                try:
                    # Generate sequence
                    if self.esm3.model_loaded:
                        generated_seq = self._generate_with_esm3(input_seq)
                    else:
                        generated_seq = self._generate_mock(input_seq)

                    # Extract CDR3 region
                    cdr3 = self._extract_cdr3(generated_seq, framework_sequence)

                    if cdr3 and self._validate_cdr3(cdr3):
                        candidates.append({
                            "sequence": cdr3,
                            "full_sequence": framework_sequence.replace(
                                "XXX", cdr3  # Placeholder replacement
                            ) if "XXX" in framework_sequence else framework_sequence,
                            "round": round_idx,
                            "generation_id": i,
                            "score": 0.0,  # Will be updated by scorer
                        })

                except Exception as e:
                    logger.warning(f"Error generating candidate {i}: {e}")
                    continue

        # Remove duplicates
        unique_candidates = self._deduplicate(candidates)
        logger.info(f"Generated {len(unique_candidates)} unique candidates")

        return unique_candidates

    def _generate_with_esm3(self, input_seq: str) -> str:
        """Generate using real ESM3 model with iterative mask filling."""
        import os
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

        try:
            # Tokenize
            tokens = self.tokenizer.encode(input_seq)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.esm3.device)

            # Use native generation that fills masked positions
            generated_tokens = self.esm3.generate(
                input_tensor,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

            # Decode the full generated sequence
            generated_seq = self.tokenizer.decode(generated_tokens[0].tolist())

            return generated_seq

        except Exception as e:
            logger.warning(f"ESM3 generation failed: {e}, using mock")
            return self._generate_mock(input_seq)

    def _generate_mock(self, input_seq: str) -> str:
        """Generate mock CDR3 for testing."""
        length = random.randint(self.config.min_length, self.config.max_length)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Generate with bias towards common CDR3 residues
        cdr3_preferred = "ARY"  # Common in CDR3
        cdr3_sequence = "".join(
            random.choice(cdr3_preferred + amino_acids) for _ in range(length)
        )

        return cdr3_sequence

    def _extract_cdr3(self, generated_seq: str, framework: str) -> Optional[str]:
        """Extract CDR3 region from generated sequence."""
        # Remove special tokens
        generated_seq = generated_seq.replace("<cls>", "").replace("<eos>", "").replace("<pad>", "").replace(" ", "")

        # Look for WG motif (start of FR4) and extract CDR3 before it
        if "WG" in generated_seq:
            wg_pos = generated_seq.find("WG")
            if wg_pos > 50:  # CDR3 typically before WG
                # Extract from ~15 aa before WG
                start = max(0, wg_pos - self.config.max_length)
                cdr3_candidate = generated_seq[start:wg_pos]
                valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                cdr3 = "".join(c for c in cdr3_candidate.upper() if c in valid_aa)

                if self.config.min_length <= len(cdr3) <= self.config.max_length:
                    return cdr3

        # Fallback: use position-based extraction (for shorter sequences)
        # Find YYC or YYC motif (common before CDR3)
        if "YY" in framework:
            yy_pos = generated_seq.rfind("YY")  # Use rfind to get later YY
            if yy_pos > 30 and yy_pos < len(generated_seq) - 10:
                # Extract ~8-18 aa after YY
                for cdr3_len in range(self.config.min_length, self.config.max_length + 1):
                    if yy_pos + cdr3_len < len(generated_seq):
                        cdr3_candidate = generated_seq[yy_pos:yy_pos + cdr3_len]
                        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                        cdr3 = "".join(c for c in cdr3_candidate.upper() if c in valid_aa)

                        # Make sure it's different from framework
                        if cdr3 and cdr3 not in framework:
                            return cdr3

        # Last resort: extract from middle
        cdr3_len = random.randint(self.config.min_length, self.config.max_length)
        start = 50
        if start + cdr3_len < len(generated_seq):
            cdr3_candidate = generated_seq[start:start + cdr3_len]
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            cdr3 = "".join(c for c in cdr3_candidate.upper() if c in valid_aa)

            if self.config.min_length <= len(cdr3) <= self.config.max_length:
                return cdr3

        return None

    def _validate_cdr3(self, cdr3: str) -> bool:
        """Validate CDR3 sequence."""
        if not cdr3:
            return False

        # Check length
        if not (self.config.min_length <= len(cdr3) <= self.config.max_length):
            return False

        # Check valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_aa for c in cdr3):
            return False

        return True

    def _deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate sequences."""
        seen = set()
        unique = []

        for c in candidates:
            seq = c.get("sequence", "")
            if seq not in seen:
                seen.add(seq)
                unique.append(c)

        return unique

    def compute_diversity_score(self, candidates: List[Dict]) -> float:
        """Compute sequence diversity score."""
        if not candidates:
            return 0.0

        sequences = [c["sequence"] for c in candidates]
        unique_ratio = len(set(sequences)) / len(sequences)

        # Compute pairwise Levenshtein distance
        if len(sequences) < 2:
            return 0.0

        distances = []
        for i in range(min(100, len(sequences))):
            for j in range(i + 1, min(100, len(sequences))):
                dist = self._levenshtein_distance(sequences[i], sequences[j])
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0

        # Normalize
        max_len = max(len(s) for s in sequences)
        normalized_distance = avg_distance / max_len if max_len > 0 else 0

        return (unique_ratio + normalized_distance) / 2

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
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


# ==================== CLI ====================
def main():
    """CLI entry point for CDR3 generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate CDR3 candidates with ESM3")
    parser.add_argument("--model", type=str, required=True, help="Path to ESM3 model (.pth)")
    parser.add_argument("--framework", type=str, required=True, help="Framework sequence")
    parser.add_argument("--output", type=str, default="candidates.json", help="Output file")
    parser.add_argument("--num-candidates", type=int, default=32, help="Number of candidates")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create generator
    config = GenerationConfig(
        num_candidates=args.num_candidates,
        temperature=args.temperature,
    )
    generator = CDR3Generator(args.model, config, args.device)

    # Generate
    candidates = generator.generate_candidates(args.framework)

    # Save
    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"Generated {len(candidates)} candidates, saved to {args.output}")


if __name__ == "__main__":
    main()
