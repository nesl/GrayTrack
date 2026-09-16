#!/bin/bash

AUTOCOMMIT="${1:-}"

xhost +local:docker

docker start -ai slow-pylot

NEW_TAG="slow-pylot:$(date +%Y%m%d-%H%M)"
MSG="auto-commit $(date -Is)"

if [[ "$AUTOCOMMIT" == "autocommit" ]]; then
  # Always commit after exit
  docker commit \
    --author "auto" \
    --message "$MSG" \
    "slow-pylot" "slow-pylot:dev"

  echo "Committed container as image: $NEW_TAG"
fi
