#!/bin/bash
#SBATCH --job-name=ollama_gpu_job  # Job name
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --time=3-00:00:00        # Max runtime (3 days)
#SBATCH --nodes=1                # Use #number physical machines
#SBATCH --ntasks=1               # 🔥 Run #number parallel python scripts when you have different settings
#SBATCH --gres=gpu:1             # Request #number GPU, when you need more control over GPU type or specific features  (A100)
#SBATCH --cpus-per-task=8        # 🔥 Assign #number CPUs per task; Match with args.processes=8; If inference is GPU-bound, having too many CPU processes won't help.

#SBATCH --mem=8GB               # Request of memory
#SBATCH --partition=gpu_h100     # Use the GPU partition
#SBATCH --array=0-6                # !!! KEY CHANGE: Create 7 tasks (0, 1, 2, 3, 4, 5, 6)

# --- 环境设置 ---
echo "作业开始于: $(date)"
echo "加载模块..."
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source ~/.bashrc
source .venv/bin/activate
# source activate worldtaskeval


echo "Using python: $(which python)"
# --- 准备工作 ---
# 定义数据和代码所在的目录


# --- Workload Calculation ---
# 定义每个任务处理的文件数量
FILES_PER_TASK=12

# !!! 核心修正: 使用SLURM_SUBMIT_DIR构建绝对路径 !!!
DATA_DIR="$SLURM_SUBMIT_DIR/data-china"
files=($DATA_DIR/*.csv)
TOTAL_FILES=${#files[@]}

# --- 调试信息 ---
echo "脚本提交目录: $SLURM_SUBMIT_DIR"
echo "数据目录路径: $DATA_DIR"
echo "总共找到 $TOTAL_FILES 个文件"

# 如果没有找到文件则提前退出
if [ $TOTAL_FILES -eq 0 ]; then
    echo "错误: 在 $DATA_DIR 中没有找到任何 .csv 文件。作业将退出。"
    exit 1
fi

# 计算此任务的起始和结束索引
START_INDEX=$((SLURM_ARRAY_TASK_ID * FILES_PER_TASK))
END_INDEX=$((START_INDEX + FILES_PER_TASK - 1))

# 稳健性检查: 确保最后一个任务的结束索引不会超过文件总数
LAST_FILE_INDEX=$((TOTAL_FILES - 1))
if (( END_INDEX > LAST_FILE_INDEX )); then
    END_INDEX=$LAST_FILE_INDEX
fi

# --- Execution Loop ---
echo "任务ID $SLURM_ARRAY_TASK_ID: 将要处理文件索引范围 $START_INDEX 到 $END_INDEX"

# 循环处理分配给此任务的文件块
for i in $(seq $START_INDEX $END_INDEX); do
    INPUT_FILE=${files[$i]}
    echo "  - 任务 $SLURM_ARRAY_TASK_ID 正在处理: ${INPUT_FILE}"
    # 调用python脚本处理单个文件
    python "$SLURM_SUBMIT_DIR/main.py" "$INPUT_FILE"
done

echo "任务ID $SLURM_ARRAY_TASK_ID: 已完成其文件块的处理。"