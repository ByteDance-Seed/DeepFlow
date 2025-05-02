EXP_NAME="deepflow_base"
UPPER_BOUND=0.01
TIME_SAMPLIG="lognormal"
DF_IDXS="6 "
ENC_TYPE=None # {dinov1-vit-b, dinov2-vit-b}
TRANS_DEPTH=12
SSL_ALIGN=false # true for SSL Alignment (REPA)

# Parse command-line arguments
for arg in "$@"
do
  case $arg in
    --ssl-align)
      SSL_ALIGN=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

DF_IDXS_LIST=$(echo "$DF_IDXS" | awk '{printf "["; for(i=1; i<=NF; i++) printf (i>1 ? ",%s" : "%s", $i); printf "]"}')
FINAL_EXP_NAME="${EXP_NAME}_trans${TRANS_DEPTH}_${TIME_SAMPLIG}_dfidx${DF_IDXS_LIST}_upper${UPPER_BOUND}_sslalign${SSL_ALIGN}_w${ENC_TYPE}"


accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 \
  --main_process_port=55773 --mixed_precision=fp16 \
  train.py \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting=${TIME_SAMPLIG} \
  --model="DeepFlow-B/2" \
  --enc-type=${ENC_TYPE} \
  --output-dir="results/deepflow" \
  --exp-name=${FINAL_EXP_NAME} \
  --data-dir="" \
  --df-idxs="${DF_IDXS_LIST}" --scale-weight 0.2 1.0 \
  --trans-depth=${TRANS_DEPTH} \
  --tg-upper-bound=${UPPER_BOUND} \
  --max-train-steps=400000 --checkpointing-steps=400000 \
  --batch-size=256 \
    $( [ "$SSL_ALIGN" = true ] && echo "--ssl-align" ) \