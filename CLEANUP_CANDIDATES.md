# CLEANUP CANDIDATES AUDIT

## Notice
> [!IMPORTANT]
> In accordance with project safety guidelines, **NO FILES HAVE BEEN DELETED**. This table documents candidates identified during the repository audit that could potentially be cleaned or removed after explicit user review and confirmation.

---

## Candidate Items for Removal / Cleaning

| File / Folder Path | Why It Appears Removable | Dependency Check | Risk Level | Recommendation |
| :--- | :--- | :--- | :---: | :--- |
| **`__pycache__/`** (in root and subdirectories) | Auto-generated compiled Python bytecode (`.pyc` files) produced during script execution. | Re-generated automatically by Python on next run. No source code depends on static `.pyc` files. | **NONE** | **Safe to remove** whenever a clean directory state is desired. |
| **`tests/__pycache__/`** | Auto-generated compiled bytecode from test runs. | Re-generated automatically by Python. | **NONE** | **Safe to remove**. |
| **`new_baseline.py`** | 22-line experimental baseline scratch script from original repository that is not imported or used by any runner or test. | Grepped repository: No imports of `new_baseline` exist across any `.py` file. | **VERY LOW** | **Archive** to `archive/legacy_code/` rather than deleting, to preserve historical reference. |
| **`PHASE12_CODE_CHANGE_PLAN.md`** | Intermediate 60-line planning scratchpad created before executing Phase 12 edits. Fully superseded by `PHASE12_EXPERIMENT_IMPLEMENTATION_AUDIT.md`. | No active runner or report references this intermediate scratch file. | **VERY LOW** | **Archive** to `archive/development_phases/` to preserve complete chronological history. |
| **`EXPERIMENT_CONFIG.md`** | Intermediate markdown reference created during Phase 11. Fully superseded by `configs/cordex_final.json` and `PHASE15_FINAL_EXPERIMENT_DESIGN.md`. | No script references this markdown file directly. | **VERY LOW** | **Archive** to `archive/development_phases/`. |
| **`results/cordex_qre_cpu/`** | Intermediate test outputs from Phase 14A and 14B ($N=3$ CPU runs). Superseded by `results/final_cordex_qre/` ($N=6$). | Not required by final runner, but represents valuable milestone validation evidence. | **MEDIUM** | **DO NOT DELETE**. Retain in `results/cordex_qre_cpu/` or move to `archive/validation_runs/` as validation evidence. |

---

## Action Plan
- **Zero deletions performed** during this pass.
- All historical reports and scripts are recommended for archiving into organized subfolders rather than deletion, ensuring complete reproducibility and thesis defense provenance.
