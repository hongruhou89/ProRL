SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m accelerate.commands.launch \
  --config_file "$PROJECT_ROOT/config/rec_config.yaml" \
  --main_process_port 16086 \
  --num_processes 4 \
  "$PROJECT_ROOT/Proactive_RL_prorl.py" \
  --dataset ml-1m \
  --config_file "$PROJECT_ROOT/config/prorl.yaml" \
  --pretrained_ckpt "$PROJECT_ROOT/ckpt/ml-1m/Aug-20-2025_23-29-662bab/Aug-20-2025_23-29-662bab.pth" \
  --mode prorl \
  --prorl_beta 1e-2 \
  --prorl_lr 1e-4 \
  --prorl_gamma 1 \
  --reward_weight_ctr 1.0 \
  --reward_weight_ioi 1.0 \
  --reward_weight_ior 1.0 \
  --prorl_epochs 50 \
