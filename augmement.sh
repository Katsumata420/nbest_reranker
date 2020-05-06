
src=$1
hyp=$2
output=$3
GPU=$4

lm_path=/cldata/cclm/cc.kenlm
bert_path=/work/katsumata/bart/rerank/score_estimate/ged-reg

featstring="EditOps(name='EditOps0'), LM('LM0', '$lm_path', normalize=False), WordPenalty(name='WordPenalty0'), BertScore('Bert0', '$bert_path', 'albert')"

CUDA_VISIBLE_DEVICES=$GPU python augmenter.py -s $src -i $hyp -o $output -f "$featstring"
