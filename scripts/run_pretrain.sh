#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "script pwd: $(pwd)"

# Bash list
scripts=(
   "$SCRIPT_DIR/Pretrain/run_ml1m_pretrain.sh"
   "$SCRIPT_DIR/Pretrain/run_steam_pretrain.sh"
   "$SCRIPT_DIR/Pretrain/run_books_pretrain.sh"
)

for script in "${scripts[@]}"; do
    echo "=========================================="
    echo "Running: $script"
    echo "Start time: $(date)"
    echo "=========================================="
    
    bash "$script"
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ Script failed with exit code: $exit_code"
        echo "Failed script: $script"
        exit $exit_code
    else
        echo "✅ Script completed successfully"
    fi
    
    echo "End time: $(date)"
    echo ""
done

echo "=========================================="
echo "All scripts finished!"
echo "=========================================="
