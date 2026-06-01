"""ESMFold2 + DoGSite3 structure prediction (no MultiGeo-DTA dependencies)."""

from __future__ import annotations

import hashlib
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

PROTEINS_PLUS_API = "https://proteins.plus/api"
DOGSITE3_REST_URL = f"{PROTEINS_PLUS_API}/dogsite3_rest"
PDB_FILES_REST_URL = f"{PROTEINS_PLUS_API}/pdb_files_rest"
DEFAULT_USER_REQUEST_RESULTS_DIR = REPO_ROOT / "outputs" / "user_request_results"
# Backward-compatible alias for imports that still reference the old name.
DEFAULT_STRUCTURE_CACHE_DIR = DEFAULT_USER_REQUEST_RESULTS_DIR


def request_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def allocate_request_results_dir(
    base: Path = DEFAULT_USER_REQUEST_RESULTS_DIR,
) -> Path:
    """Create a timestamped directory for one user inference request."""
    base.mkdir(parents=True, exist_ok=True)
    stamp = request_timestamp()
    candidate = base / stamp
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    suffix = 1
    while (base / f"{stamp}_{suffix}").exists():
        suffix += 1
    work_dir = base / f"{stamp}_{suffix}"
    work_dir.mkdir(parents=True)
    return work_dir


def sequence_cache_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def esmfold_mmcif_to_pdb(cif_path: Path, pdb_path: Path) -> Path:
    """Convert ESMFold2 mmCIF to PDB without BioPython (mmCIF omits occupancy)."""
    records: list[list[str]] = []
    in_atom_site = False
    with cif_path.open() as handle:
        for line in handle:
            if line.startswith("loop_"):
                in_atom_site = False
            elif line.startswith("_atom_site."):
                in_atom_site = True
            elif in_atom_site and line.startswith(("ATOM", "HETATM")):
                records.append(line.split())
    if not records:
        raise ValueError(f"No atom records found in {cif_path}")

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with pdb_path.open("w") as out:
        out.write("REMARK   ESMFold2 mmCIF converted to PDB\n")
        for cols in records:
            serial = int(cols[18])
            alt = " " if cols[3] == "." else cols[3][0]
            resname, chain, resseq = cols[10], cols[11], int(cols[9])
            icode = " " if cols[8] == "." else cols[8][0]
            name, elem, bfac = cols[12], cols[1], float(cols[13])
            x, y, z = map(float, cols[14:17])
            name_fmt = (f" {name:>3}" if len(elem) == 1 and len(name) <= 3 else f"{name:>4}")[:4]
            out.write(
                f"{cols[0]:6}{serial:5d} {name_fmt}{alt}{resname:>3} {chain:1}{resseq:4d}{icode:1}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bfac:6.2f}          {elem:>2}\n"
            )
    return pdb_path


def _resolve_cached_protein_pdb(work_dir: Path, cache_key: str) -> Optional[Path]:
    """Use PDB cache, or convert a leftover mmCIF from current or legacy cache dirs."""
    cached_pdb = work_dir / "esmfold_protein.pdb"
    if cached_pdb.exists():
        return cached_pdb

    legacy_dirs = (
        REPO_ROOT / "outputs" / "structure_cache" / cache_key,
        REPO_ROOT / "scripts" / "outputs" / "structure_cache" / cache_key,
    )
    for cache_dir in (work_dir, *legacy_dirs):
        cached_cif = cache_dir / "esmfold_protein.cif"
        if cached_cif.exists():
            print(f"Converting cached mmCIF to PDB: {cached_cif}", file=sys.stderr)
            return esmfold_mmcif_to_pdb(cached_cif, cached_pdb)
    return None


def predict_protein_structure_esmfold2(
    sequence: str,
    output_pdb: Path,
    *,
    device_id: int = 0,
    num_loops: int = 3,
    num_sampling_steps: int = 50,
    seed: int = 0,
) -> Path:
    """
    Predict an all-atom protein structure with ESMFold2 (Biohub/esm).

    Requires: pip install esm@git+https://github.com/Biohub/esm.git@main
    Model weights: biohub/ESMFold2 on Hugging Face.
    """
    try:
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
        )
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except ImportError as exc:
        raise ImportError(
            "ESMFold2 requires the esm package and transformers with ESMFold2 support.\n"
            "Install with:\n"
            "  pip install esm@git+https://github.com/Biohub/esm.git@main\n"
            "  pip install transformers\n"
            "Ensure Hugging Face access to biohub/ESMFold2."
        ) from exc

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESMFold2 on {device}...", file=sys.stderr)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=sequence)]
    )
    print(
        f"Running ESMFold2 (len={len(sequence)}, loops={num_loops}, "
        f"steps={num_sampling_steps})...",
        file=sys.stderr,
    )
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=1,
        seed=seed,
    )
    plddt_mean = float(result.plddt.mean())
    print(
        f"ESMFold2 finished: pLDDT={plddt_mean:.3f}, pTM={float(result.ptm):.3f}",
        file=sys.stderr,
    )

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    # MolecularComplex has to_mmcif() but not to_pdb(); convert via ProteinComplex.
    result.complex.to_protein_complex().to_pdb(output_pdb)
    return output_pdb


def _poll_proteins_plus_job(
    job_url: str,
    *,
    poll_interval: float = 10.0,
    max_attempts: int = 60,
    label: str = "job",
) -> dict:
    """Poll a ProteinsPlus REST job until status_code 200."""
    for attempt in range(max_attempts):
        response = requests.get(job_url, headers={"Accept": "application/json"}, timeout=120)
        response.raise_for_status()
        data = response.json()
        status = data.get("status_code")
        if status == 200:
            return data
        if status == 429:
            print("  Rate limited by ProteinsPlus; retrying...", file=sys.stderr)
            time.sleep(poll_interval * 2)
            continue
        if status not in (202, "accepted", "202"):
            raise RuntimeError(f"Unexpected {label} response from {job_url}: {data}")
        print(f"  Waiting for {label} ({attempt + 1}/{max_attempts})...", file=sys.stderr)
        time.sleep(poll_interval)
    raise TimeoutError(f"{label} did not finish in time: {job_url}")


def upload_pdb_to_proteins_plus(pdb_path: Path) -> str:
    """Upload a custom PDB to ProteinsPlus and return its API identifier."""
    print(f"Uploading {pdb_path.name} to ProteinsPlus...", file=sys.stderr)
    with pdb_path.open("rb") as handle:
        response = requests.post(
            PDB_FILES_REST_URL,
            files={"pdb_file[pathvar]": (pdb_path.name, handle, "chemical/x-pdb")},
            headers={"Accept": "application/json"},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    location = data.get("location")
    if not location:
        raise RuntimeError(f"PDB upload failed: {data}")

    for attempt in range(60):
        poll = requests.get(location, headers={"Accept": "application/json"}, timeout=60)
        poll.raise_for_status()
        pdata = poll.json()
        pdb_id = pdata.get("id")
        message = str(pdata.get("message", "")).lower()
        if pdb_id and "loaded" in message:
            print(f"PDB uploaded as id={pdb_id}", file=sys.stderr)
            return pdb_id
        time.sleep(2)
        if attempt % 5 == 4:
            print(f"  Waiting for PDB upload ({attempt + 1}/60)...", file=sys.stderr)
    raise TimeoutError(f"PDB upload did not complete: {location}")


def submit_dogsite3_job(
    pdb_code: str,
    *,
    chain: str = "",
    ligand: str = "",
    ligand_bias: bool = False,
    analysis_detail: str = "0",
    binding_site_prediction_granularity: str = "0",
) -> str:
    """Submit a DoGSite3 job and return the job URL."""
    payload = {
        "dogsite3": {
            "pdbCode": pdb_code,
            "analysisDetail": analysis_detail,
            "bindingSitePredictionGranularity": binding_site_prediction_granularity,
            "ligand": ligand,
            "chain": chain,
            "ligandBias": "1" if ligand_bias else "0",
        }
    }
    response = requests.post(
        DOGSITE3_REST_URL,
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    location = data.get("location")
    if not location:
        raise RuntimeError(f"DoGSite3 submission failed: {data}")
    print(f"DoGSite3 job submitted: {location}", file=sys.stderr)
    return location


def select_best_dogsite3_pocket(
    result_table_text: str,
    pocket_id: Optional[str] = None,
) -> str:
    """Pick the top-ranked pocket from a DoGSite3 descriptor table."""
    table = pd.read_csv(io.StringIO(result_table_text), sep="\t").set_index("name")
    if pocket_id is not None:
        if pocket_id not in table.index:
            available = ", ".join(map(str, table.index))
            raise ValueError(f"Pocket {pocket_id!r} not found. Available: {available}")
        return pocket_id
    if (table["lig_cov"] > 0).any():
        return str(table.sort_values(["lig_cov", "poc_cov"], ascending=False).index[0])
    return str(table.sort_values("volume", ascending=False).index[0])


def _dogsite3_pocket_residue_url(job_result: dict, pocket_id: str) -> str:
    pocket_token = f"{pocket_id}_res"
    for url in job_result.get("residues", []):
        if pocket_token in url:
            return url
    raise ValueError(f"No residue PDB found for pocket {pocket_id!r}")


def predict_pocket_structure_dogsite3(
    protein_pdb: Path,
    output_pocket_pdb: Path,
    *,
    chain: str = "",
    ligand: str = "",
    ligand_bias: bool = False,
    pocket_id: Optional[str] = None,
    poll_interval: float = 10.0,
    max_attempts: int = 60,
) -> Tuple[Path, str]:
    """Run DoGSite3 on a protein PDB and download the selected pocket residue PDB."""
    uploaded_id = upload_pdb_to_proteins_plus(protein_pdb)
    job_url = submit_dogsite3_job(
        uploaded_id,
        chain=chain,
        ligand=ligand,
        ligand_bias=ligand_bias,
    )
    job_result = _poll_proteins_plus_job(
        job_url,
        poll_interval=poll_interval,
        max_attempts=max_attempts,
        label="DoGSite3",
    )

    result_table_url = job_result["result_table"]
    result_table_text = requests.get(result_table_url, timeout=120).text
    selected_pocket = select_best_dogsite3_pocket(result_table_text, pocket_id=pocket_id)
    print(f"Selected DoGSite3 pocket: {selected_pocket}", file=sys.stderr)

    pocket_url = _dogsite3_pocket_residue_url(job_result, selected_pocket)
    pocket_response = requests.get(pocket_url, timeout=120)
    pocket_response.raise_for_status()
    output_pocket_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pocket_pdb.write_text(pocket_response.text)
    return output_pocket_pdb, selected_pocket


def ensure_structure_files(
    protein_sequence: str,
    *,
    protein_pdb: Optional[Path],
    pocket_pdb: Optional[Path],
    request_results_dir: Optional[Path],
    device_id: int = 0,
    force_repredict: bool = False,
    esmfold_num_loops: int = 3,
    esmfold_num_sampling_steps: int = 50,
    esmfold_seed: int = 0,
    dogsite_chain: str = "",
    dogsite_ligand: str = "",
    dogsite_ligand_bias: bool = False,
    dogsite_pocket_id: Optional[str] = None,
    dogsite_poll_interval: float = 10.0,
    dogsite_max_attempts: int = 60,
) -> Tuple[Path, Path, Dict]:
    """Ensure full protein and pocket PDB files exist, running ESMFold2 / DoGSite3 if needed."""
    metadata: Dict = {}
    work_dir = request_results_dir or allocate_request_results_dir()
    work_dir.mkdir(parents=True, exist_ok=True)

    resolved_protein = protein_pdb
    if resolved_protein is None:
        cache_key = sequence_cache_key(protein_sequence)
        cached_protein = work_dir / "esmfold_protein.pdb"
        if not force_repredict:
            cached = _resolve_cached_protein_pdb(work_dir, cache_key)
            if cached is not None:
                print(f"Using cached ESMFold2 structure: {cached}", file=sys.stderr)
                resolved_protein = cached
                metadata["protein_source"] = "cache"
        if resolved_protein is None:
            resolved_protein = predict_protein_structure_esmfold2(
                protein_sequence,
                cached_protein,
                device_id=device_id,
                num_loops=esmfold_num_loops,
                num_sampling_steps=esmfold_num_sampling_steps,
                seed=esmfold_seed,
            )
            metadata["protein_source"] = "esmfold2"
        metadata["protein_pdb"] = str(resolved_protein)
    else:
        metadata["protein_source"] = "user"
        metadata["protein_pdb"] = str(resolved_protein)

    resolved_pocket = pocket_pdb
    if resolved_pocket is None:
        cached_pocket = work_dir / "dogsite3_pocket.pdb"
        if cached_pocket.exists() and not force_repredict:
            print(f"Using cached DoGSite3 pocket: {cached_pocket}", file=sys.stderr)
            resolved_pocket = cached_pocket
            metadata["pocket_source"] = "cache"
        else:
            resolved_pocket, selected_pocket = predict_pocket_structure_dogsite3(
                resolved_protein,
                cached_pocket,
                chain=dogsite_chain,
                ligand=dogsite_ligand,
                ligand_bias=dogsite_ligand_bias,
                pocket_id=dogsite_pocket_id,
                poll_interval=dogsite_poll_interval,
                max_attempts=dogsite_max_attempts,
            )
            metadata["pocket_source"] = "dogsite3"
            metadata["dogsite_pocket_id"] = selected_pocket
        metadata["pocket_pdb"] = str(resolved_pocket)
    else:
        metadata["pocket_source"] = "user"
        metadata["pocket_pdb"] = str(resolved_pocket)

    metadata["request_results_dir"] = str(work_dir)
    metadata["request_timestamp"] = work_dir.name
    return resolved_protein, resolved_pocket, metadata
