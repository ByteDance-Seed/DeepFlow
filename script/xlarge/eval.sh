EXP_NAME="deepflow_xlarge"
UPPER_BOUND=0.01
TIME_SAMPLIG="lognormal"
DF_IDXS="8 16"
ENC_TYPE=None # {dinov1-vit-b, dinov2-vit-b}
TRANS_DEPTH=24
SSL_ALIGN=false # true for SSL Alignment (REPA)
LEGACY_SCALING=false
CFG_SCALE=1.0

# Parse command-line arguments
for arg in "$@"
do
  case $arg in
    --ssl-align)
      SSL_ALIGN=true
      shift
      ;;
    --legacy-scaling)
      LEGACY_SCALING=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

DF_IDXS_LIST=$(echo "$DF_IDXS" | awk '{printf "["; for(i=1; i<=NF; i++) printf (i>1 ? ",%s" : "%s", $i); printf "]"}')
FINAL_EXP_NAME="{CKPT_NAME}"


torchrun --nnodes=1 --nproc_per_node=8 --master_port 12333 generate.py \
  --model="DeepFlow-XL/2" \
  --ckpt results/${FINAL_EXP_NAME}/checkpoints/2000000.pt \
  --num-fid-samples 50000 \
  --path-type=linear \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=${CFG_SCALE} \
  --sample-dir="samples/${FINAL_EXP_NAME}" \
  --trans-depth=${TRANS_DEPTH} \
  --df-idxs="${DF_IDXS_LIST}" \
    $( [ "$SSL_ALIGN" = true ] && echo "--ssl-align" ) \
    $( [ "$LEGACY_SCALING" = true ] && echo "--legacy-scaling" ) \


