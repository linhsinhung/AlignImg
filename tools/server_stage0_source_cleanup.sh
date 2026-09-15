#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---scan}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [[ "$mode" != "--scan" && "$mode" != "--apply" ]]; then
  echo "Usage: bash tools/server_stage0_source_cleanup.sh [--scan|--apply]" >&2
  exit 2
fi

if [[ ! -f "$repo_root/pyproject.toml" || ! -d "$repo_root/src/alignimg" ]]; then
  echo "Refusing to continue: repository root could not be verified: $repo_root" >&2
  exit 2
fi

stale_paths=(
  "examples/align_single_demo.py"
  "examples/benchmark_cpu_optimization.py"
  "examples/canonical_math_demo.py"
  "examples/realdata_demo.py"
  "src/alignimg/_batch_cpu.py"
  "src/alignimg/_multicore.py"
  "src/alignimg/_utils.py"
  "src/alignimg/api.py"
  "tests/test_alignimg_canonical_math.py"
  "tests/test_batch_scan_cpu.py"
  "tests/test_multicore_optional.py"
)

found=()
for relative in "${stale_paths[@]}"; do
  if [[ -e "$repo_root/$relative" ]]; then
    found+=("$relative")
  fi
done

echo "Repository: $repo_root"
echo "Known stale paths found: ${#found[@]}"
if (( ${#found[@]} > 0 )); then
  for relative in "${found[@]}"; do
    echo "  $relative"
  done
fi

if [[ "$mode" == "--scan" ]]; then
  echo "Dry run only. Re-run with --apply to move these paths into validation-results/drop/."
  exit 0
fi

if (( ${#found[@]} > 0 )); then
  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  drop_root="$repo_root/validation-results/drop/stage-0-source-cleanup-$timestamp"
  if [[ -e "$drop_root" ]]; then
    echo "Refusing to overwrite existing drop directory: $drop_root" >&2
    exit 2
  fi
  mkdir -p "$drop_root"
  for relative in "${found[@]}"; do
    destination="$drop_root/$relative"
    mkdir -p "$(dirname "$destination")"
    mv "$repo_root/$relative" "$destination"
  done
  echo "Moved ${#found[@]} paths to: $drop_root"
else
  echo "No known stale paths needed moving."
fi

notice_sha="$(sha256_file "$repo_root/THIRD_PARTY_NOTICES.md")"
expected_notice_sha="ac982b163aa2a67dbb9cc7a52320f89e0d87e8885bfb9cf05b682bc42d69e5f6"
if [[ "$notice_sha" != "$expected_notice_sha" ]]; then
  echo "THIRD_PARTY_NOTICES.md does not match the local release source." >&2
  echo "Expected: $expected_notice_sha" >&2
  echo "Actual:   $notice_sha" >&2
  echo "Upload the current local file in binary mode, then run this script again." >&2
  exit 1
fi

validator_sha="$(sha256_file "$repo_root/tools/performance_validation.py")"
expected_validator_sha="3d844bc98d1e9d197b3fc262310fde59c1b15102f7dcfd78ccd34ae1ef272f7e"
if [[ "$validator_sha" != "$expected_validator_sha" ]]; then
  echo "tools/performance_validation.py is not the accepted 5e-6 validator." >&2
  echo "Expected: $expected_validator_sha" >&2
  echo "Actual:   $validator_sha" >&2
  exit 1
fi

source_sha="$(cd "$repo_root" && python - <<'PY'
from tools.performance_fixtures import source_manifest

print(source_manifest()["source_sha256"])
PY
)"
expected_source_sha="175176b13c4a19aaf39c72fe88e11eaf893983302f304cf5d71f639f3cdaad62"
if [[ "$source_sha" != "$expected_source_sha" ]]; then
  echo "Source cleanup is incomplete; checkout still differs from the local release source." >&2
  echo "Expected: $expected_source_sha" >&2
  echo "Actual:   $source_sha" >&2
  exit 1
fi

echo "Source verification passed: $source_sha"
