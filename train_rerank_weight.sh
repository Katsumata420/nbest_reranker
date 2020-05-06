
input=$1
ref=$2
config=$3
output_dir=$4

moses_path=/clwork/katsumata/mosesdecoder

mkdir -p $output_dir

python train.py -i $input -r $ref -c $config --threads 12 --tuning-metric m2 --predictable-seed -o $output_dir --moses-dir $moses_path --no-add-weight
