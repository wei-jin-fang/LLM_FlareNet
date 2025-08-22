#!/bin/bash

# Get the absolute path of the directory where this script is located
# 因为后面很多都是基于相对路径进行创建文件的
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
echo "Current directory: $(pwd)"

# 全局变量配置 - 从train.py的read_parameters函数提取的参数
CONDA_ENV="VGGT_py310"  # 请修改为你的conda环境名称
NUM_DATASET=9
INPUT_SIZE=10
TIME_STEP=40
BATCH_SIZE=16
DEVICE="cuda"
DATASETNAME="data"
EPOCHS=50
OPTIM="Adam"
SEED=2021
# LLMFlareNetModel训练参数
BERT_EMB=768 #不可以修改
D_LLM=768    #不可以修改
D_MODEL=16 #可以修改，这个是patch完成的维度，用于参与att，但是要注意和head个数乘倍数整除
BERT_NUM_HIDDEN_LAYERS=2  #可以修改，但是越大大模型越大，时间越慢，越复杂
DESCRIPTION_DATA="数据形状是40*10,由40个耀斑物理特征时间步数据组成，每个时间步有10个特征,每一组数据分别对应未来24小时内爆发的耀斑类别是否是大于等于M类别" #可以修改，换成英文最好，表述没有固定答案
DESCRIPTION_TASK="使用这些数据预报未来24小时内爆发大于等于M类别耀斑的概率,预报的概率值大于0.5则视为发生了" #可以修改，换成英文最大，表述没有固定答案
N_HEADS=8 #可以修改，注意力机制头，但是要注意和D_MODEL是倍数
DROPOUT=0.5  #可以修改，
NUM_TOKENS=1000  #可以修改，预训练权重变映射成多少个词
PATCH_LEN=1 #暂时不要修改保证1*10patch
STRIDE=1    #暂时不要修改保证1*10patch

# OnefitallModel训练参数
#没啥好修改的参数，DROPOUT=0.5  #可以修改，公用的，pathc里面的参数

# 输出层训练参数
BATCH_NORM64_DIM=64  #可以修改，但是要保证和FC64_DIM一样
BATCH_NORM32_DIM=32  #可以修改，但是要保证和BATCH_NORM32_DIM一样
DROPOUT_RATE=0.5  #可以修改，输出层的
FC64_DIM=64  #可以修改，但是要保证和BATCH_NORM64_DIM一样
FC32_DIM=32  #可以修改，但是要保证和BATCH_NORM32_DIM一样
OUTPUT_DIM=1  #不要修改，sigmod输出是1
COMMENT="None" #不用管

# 需要调参的参数配置 - 修改这里来指定要调参的参数
# 示例：调整学习率 LR 参数，如果要调这个，手动修改下run_training() 里面的    local current_lr=$2 只需要修改等号左边的，然后放到下面脚本里面对应位置即可，记的
#一定要记得对于一个参数循环调，要修改49-52行（4处），61行一处， 函数里面参数替换成你要改的那个比如batch_size就要修改66行

PARAM_NAME="LR"  # 要调整的参数名称，这个名字是用于打印的，不影响跳槽那
PARAM_START=0.001  # 起始值
PARAM_STEP=0.001   # 步长  公式=0.001+0.001*n ，n就是循环多少次从0-4
PARAM_NUM=2        # 循环次数

# 激活conda环境
echo "Activating conda environment: $CONDA_ENV"
source activate $CONDA_ENV

# 运行训练函数
run_training() {
    local model_type=$1
    local current_lr=$2  #
    
    echo "Running training with model: $model_type, LR: $current_lr"
    python train.py \
        --num_dataset $NUM_DATASET \
        --input_size $INPUT_SIZE \
        --time_step $TIME_STEP \
        --batch_size $BATCH_SIZE \
        --device $DEVICE \
        --datasetname $DATASETNAME \
        --epochs $EPOCHS \
        --lr $current_lr \
        --optim $OPTIM \
        --seed $SEED \
        --model_type $model_type \
        --bert_emb $BERT_EMB \
        --d_llm $D_LLM \
        --d_model $D_MODEL \
        --bert_num_hidden_layers $BERT_NUM_HIDDEN_LAYERS \
        --description_data "$DESCRIPTION_DATA" \
        --description_task "$DESCRIPTION_TASK" \
        --n_heads $N_HEADS \
        --dropout $DROPOUT \
        --num_tokens $NUM_TOKENS \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --batch_norm64_dim $BATCH_NORM64_DIM \
        --batch_norm32_dim $BATCH_NORM32_DIM \
        --dropout_rate $DROPOUT_RATE \
        --fc64_dim $FC64_DIM \
        --fc32_dim $FC32_DIM \
        --output_dim $OUTPUT_DIM \
        --conmment "${model_type}_${PARAM_NAME}_${current_lr}"
}

# 主循环 - 对指定参数进行调参
echo "开始对参数 $PARAM_NAME 进行调参，共执行 $PARAM_NUM 次..."

for i in $(seq 1 $PARAM_NUM); do
    # 计算当前参数值
    current_param_value=$(echo "$PARAM_START + ($i - 1) * $PARAM_STEP" | bc -l)
    
    echo "=== 第 $i 次调参，$PARAM_NAME = $current_param_value ==="
    
    # 运行指定的模型（可以修改这里选择要调参的模型）
    run_training "Onefitall_11" $current_param_value
    run_training "Onefitall_12" $current_param_value
    run_training "Onefitall_13" $current_param_value
    run_training "LLMFlareNet_1" $current_param_value
    run_training "LLMFlareNet_2" $current_param_value
    
    echo "第 $i 次调参完成"
    echo "================================="
done

echo "所有调参完成！"
# LR 进行调参，共