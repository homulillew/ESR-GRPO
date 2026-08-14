#!/usr/bin/env bash
set -euo pipefail

: "${ECHO_ROOT:?set ECHO_ROOT}"
BASELINE="${1:-}"
case "${BASELINE}" in
  grpo)
    SCRIPT="${ECHO_ROOT}/examples/sglang_multiturn/run_qwen3-30b-a3b_bcp_grpo_fully_async_4node.sh"
    ;;
  echo)
    SCRIPT="${ECHO_ROOT}/examples/sglang_multiturn/run_qwen3-30b-a3b_bcp_echo-ca_fully_async_4node.sh"
    ;;
  *)
    echo "usage: $0 {grpo|echo}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${SCRIPT}" ]]; then
  echo "missing upstream ECHO script: ${SCRIPT}" >&2
  exit 1
fi
echo "This baseline uses the upstream 4-node x 8-GPU resource layout: ${SCRIPT}"
exec bash "${SCRIPT}"
