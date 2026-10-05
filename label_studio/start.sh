#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/ls_data"
chmod 777 "$DIR/ls_data" 2>/dev/null || true

echo "Starting Label Studio container..."
cd "$DIR"
docker compose up -d

echo "=========================================="
echo "Label Studio is running!"
echo "URL: http://localhost:8080"
echo "=========================================="
