#!/usr/bin/env bash
set -euo pipefail
archive_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="$archive_root/race90/competition/q1/models-selected.tgz"
parts="$output.parts"
expected_sha256='8fdede36d3eeaf7161c6d6f2897ec104d15923fde4276ff0d190bb796b007b07'
expected_bytes='3056759127'
if [[ -e "$output" ]]; then
  if [[ "$(sha256sum -- "$output" | cut -d ' ' -f 1)" == "$expected_sha256" ]]; then
    echo "Already restored and verified: $output"
    exit 0
  fi
  echo "Existing output differs; refusing to overwrite: $output" >&2
  exit 1
fi
(cd -- "$parts" && sha256sum --check SHA256SUMS)
temporary="$output.restoring.$$"
trap 'rm -f -- "$temporary"' EXIT
cat -- "$parts"/part-[0-9][0-9][0-9][0-9] > "$temporary"
if [[ "$(wc -c < "$temporary" | tr -d '[:space:]')" != "$expected_bytes" ]]; then
  echo "Restored file size mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum -- "$temporary" | cut -d ' ' -f 1)" != "$expected_sha256" ]]; then
  echo "Restored file SHA256 mismatch" >&2
  exit 1
fi
mv -- "$temporary" "$output"
trap - EXIT
printf 'Restored and verified: %s\nSHA256: %s\n' "$output" "$expected_sha256"
