#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 RAW_SOURCE_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_SOURCE_DIR="$(cd "$1" && pwd)"
NEW_OUTPUT_DIR="$2"

if [[ -e "${NEW_OUTPUT_DIR}" ]]; then
  echo "Output path already exists; choose a new directory: ${NEW_OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${NEW_OUTPUT_DIR}"
NEW_OUTPUT_DIR="$(cd "${NEW_OUTPUT_DIR}" && pwd)"

case "${NEW_OUTPUT_DIR}/" in
  "${PACKAGE_DIR}"/*)
    echo "Refusing to write audit output inside the package directory" >&2
    exit 1
    ;;
esac

CAMSOL_SUPP="${RAW_SOURCE_DIR}/CamSolPTM_Supplementary_Data_1.xlsx"
CAMSOL_SOURCE="${RAW_SOURCE_DIR}/CamSolPTM_Source_Data.xlsx"
STAPLED_SOURCE="${RAW_SOURCE_DIR}/StapledPeptides_Supplementary_Dataset_1.xlsx"
PEGASUS_ROOT="${RAW_SOURCE_DIR}/PEGASUS1"
PEGASUS_SOURCE="${PEGASUS_ROOT}/inhouse_design.csv"

for source_file in \
  "${CAMSOL_SUPP}" \
  "${CAMSOL_SOURCE}" \
  "${STAPLED_SOURCE}" \
  "${PEGASUS_SOURCE}"; do
  if [[ ! -f "${source_file}" ]]; then
    echo "Required source file not found: ${source_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/audit_camsolptm_endpoint.py" \
  "${CAMSOL_SUPP}" "${CAMSOL_SOURCE}" \
  --root "${RAW_SOURCE_DIR}" \
  --output "${NEW_OUTPUT_DIR}/camsolptm_endpoint_audit_20260804.json"

"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/audit_stapled_solubility_dataset.py" \
  "${STAPLED_SOURCE}" \
  --root "${RAW_SOURCE_DIR}" \
  --output "${NEW_OUTPUT_DIR}/stapled_peptides_endpoint_audit_20260804.json"

# The normalised PEGASUS table is row-level third-party data.  It is created in
# a private temporary directory and removed once the aggregate audit is written,
# so no row-level file is ever left in a caller-chosen output directory.
ROW_LEVEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stam_m_rowlevel.XXXXXX")"
trap 'rm -rf "${ROW_LEVEL_DIR}"' EXIT

"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/normalize_public_csv_headers.py" \
  "${PEGASUS_SOURCE}" \
  "${ROW_LEVEL_DIR}/inhouse_design_values_only.csv" \
  --root "${PEGASUS_ROOT}" \
  --manifest "${NEW_OUTPUT_DIR}/inhouse_design_transform.json"

"${PYTHON_BIN}" "${PACKAGE_DIR}/scripts/audit_pegasus_solubility.py" \
  "${ROW_LEVEL_DIR}/inhouse_design_values_only.csv" \
  --root "${ROW_LEVEL_DIR}" \
  --repository-commit "5df3a4b073181b08175af16dc7e0526aa5e34179" \
  --output "${NEW_OUTPUT_DIR}/pegasus_endpoint_audit_20260804.json"

rm -rf "${ROW_LEVEL_DIR}"
trap - EXIT

echo "Audit regeneration complete: ${NEW_OUTPUT_DIR}"
