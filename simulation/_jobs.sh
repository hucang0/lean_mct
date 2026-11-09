#!/usr/bin/env bash
# Shared by the production launchers. Wait for every child and propagate failures.
job_pids=()
job_labels=()

register_job() {
  job_pids+=("$1")
  job_labels+=("$2")
}

wait_jobs() {
  local status=0 i
  for i in "${!job_pids[@]}"; do
    if wait "${job_pids[$i]}"; then
      :
    else
      printf 'FAILED: %s (pid %s); see logs/\n' "${job_labels[$i]}" "${job_pids[$i]}" >&2
      status=1
    fi
  done
  if [ "$status" -eq 0 ]; then
    printf 'Completed all %s jobs.\n' "${#job_pids[@]}"
  fi
  return "$status"
}
