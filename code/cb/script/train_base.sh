
dataset="mydataset11"

train_data_path="./data/${dataset}/train.jsonl"
valid_data_path="./data/${dataset}/valid.jsonl"
test_data_path="./data/${dataset}/test.jsonl"

test=0


dataset_type="BaseDataset"
model_type="BaseCodeBert"
checkpoint="0914-1"
output_dir="./output_dir/${dataset}_${dataset_type}_${model_type}"
mkdir -p ${output_dir}
log_dir="./log/${dataset}_${dataset_type}_${model_type}.log"


device=3
train_batch_size=32
valid_batch_size=32
max_input_len=256
max_output_len=128
num_train_epochs=20
learning_rate=5e-5
adam_epsilon=1e-8

nohup python train_cb.py    \
    --test ${test}  \
    --device ${device}  \
    --train_data_path ${train_data_path}    \
    --valid_data_path ${valid_data_path}    \
    --test_data_path ${test_data_path}  \
    --output_dir ${output_dir}  \
    --dataset_type ${dataset_type}  \
    --model_type ${model_type}  \
    --checkpoint ${checkpoint}  \
    --max_input_len ${max_input_len}  \
    --max_output_len ${max_output_len}  \
    --num_train_epochs ${num_train_epochs}    \
    --train_batch_size ${train_batch_size}    \
    --valid_batch_size ${valid_batch_size} \
    --learning_rate ${learning_rate}    \
    --adam_epsilon ${adam_epsilon}  \
    > ${log_dir} 2>&1 &
