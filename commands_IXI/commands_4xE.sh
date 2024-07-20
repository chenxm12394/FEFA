PWD=/data2/cxm/dataset
TRAIN="${PWD}/IXI/pd_t2_paired_IXI_train.csv"
VAL="${PWD}/IXI/pd_t2_paired_IXI_val.csv"
LOG_BASE="./LOG_IXI"
mkdir -p ${LOG_BASE}
COILS=1
TGT=T2
REF=PD
FLAGS='--prefetch --force_gpu'
export CUDA_VISIBLE_DEVICES=1
mkdir -p ${LOG_BASE}


# Training

NAME=4xEquispaced_bb
MASK=equispaced
SPAR=0.25

# NAME=8xEquispaced_bb
# MASK=equispaced
# SPAR=0.125

# NAME=16xEquispaced_bb
# MASK=equispaced
# SPAR=0.0625

# NAME=4xRandom_bb
# MASK=standard
# SPAR=0.25

# NAME=8xRandom_bb
# MASK=standard
# SPAR=0.125

# NAME=16xRandom_bb
# MASK=standard
# SPAR=0.0625

# /home/cxm/miniconda3/envs/MMMRI/bin/python train.py --logdir ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineProposed --train ${TRAIN} --val ${VAL} --num_workers 2 --lr 1e-4 --sim_weight 1 --protocals ${TGT} ${REF} --mask ${MASK} --aux_aug PBSpline --sparsity ${SPAR} --epoch 1000 --batch_size 4 --reg None  --intel_stop 2e4 --indexs 1 --crop 256 --coils ${COILS}  ${FLAGS}


# Testing
EVAL_BASE="${PWD}/eval/IXI/Ours"
DATA_TEST="${PWD}/IXI/pd_t2_paired_IXI_test.csv"
AUX_AUG='-1'

function run_test(){
  echo ${NAME}
  mkdir -p ${EVAL_BASE}/${NAME}
  if test -f ${EVAL_BASE}/${NAME}/md5sum && md5sum -c ${EVAL_BASE}/${NAME}/md5sum
  then
    echo SKIPPED
  else
    /home/cxm/miniconda3/envs/MMMRI/bin/python eval.py \
      --resume ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineProposed/ckpt/best.pt \
      --val ${DATA_TEST} \
      --protocals ${PROTOCALS} --aux_aug ${AUX_AUG} \
      --save ${EVAL_BASE}/${NAME} \
      --metric ${EVAL_BASE}/${NAME}.json\
      --indexs 1 \
      --crop 256
    md5sum ${LOG_BASE}/${NAME}/ckpt/best.pt/* > ${EVAL_BASE}/${NAME}/md5sum
  fi
}

# # # # Single-Modal
# # # PROTOCALS="${TGT} None"
# # # ENAME="None_${NAME}${TGT}_PBSplineNone" run_test
# # # # Multi-Modal
PROTOCALS="${TGT} ${REF}"
# # # ENAME="${REF}_${NAME}${TGT}_PBSplineNone" run_test
# # # # Proposed
ENAME="${REF}_${NAME}${TGT}_PBSplineProposed" run_test

