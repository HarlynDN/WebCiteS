# WebCiteS: Attributed Query-Focused Summarization on Chinese Web Search Results with Citations

本项目提供了论文[WebCiteS: Attributed Query-Focused Summarization on Chinese Web Search Results with Citations](https://arxiv.org/abs/2403.01774)的代码实现。在这个工作中，我们形式化了AQFS任务并提出了WebCiteS数据集。WebCiteS来源于真实的用户查询和网络搜索数据，包含7k条人工参与标注的引证式摘要。我们针对摘要质量和结果的可归因性（attribution）进行了全面评估，同时还基于开源模型构建了一个高效的自动评估器。请参阅论文来获取更多细节。


## 目录
- [设置](#设置)
- [准备评估器](#准备评估器)
- [进行AQFS任务的实验](#进行aqfs任务的实验)

## 设置
首先，克隆本仓库并进入根目录。
```bash
git clone https://github.com/HarlynDN/WebCiteS.git
cd WebCiteS
```
### 配置环境
```bash
pip install -r requirements.txt
```

### 准备数据
运行下面的命令来下载并预处理数据集。
```bash
wget https://huggingface.co/datasets/HarlynDN/WebCiteS/resolve/main/data.tar
tar -xvf data.tar
echo "deleting tar file"
rm data.tar
cd data
echo "preprocessing the data"
python prepare_data.py --raw_data webcites.json
cd ..
```

## 准备评估器

### 训练观点分解模型
输入一句话，观点分解模型会生成其所包含的所有子观点。下面的命令将会微调`google/mt5-large`模型来得到观点分解模型。

```bash
cd src
NUM_GPUs=8
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUs claimsplit/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir ../data/claimsplit \
    --source_column sentence \
    --target_column claims \
    --overwrite_cache \
    --model_name_or_path google/mt5-large \
    --output_dir {path/to/output/directory} \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --predict_with_generate 
```

### 评估观点分解模型的性能

我们使用一个NLI模型来评估生成的子观点的冗余性、正确性和完整性。在本工作中，我们使用了`alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli`这一NLI模型。

```bash
cd src
python claimsplit/eval.py \
    --pred_file {path/to/claimsplit/generation/file} \
    --nli_model_path alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli \
    --batch_size 256
```
如果遇到OOM错误，尝试减小`batch_size`。


### 评估完整评估器的性能

完整的评估器包含一个观点分解模型和一个NLI模型，我们在WebCiteS的测试集上评估整个评估器的性能。


```bash
cd src
python aqfs/eval_evaluator.py \
    --test_file ../data/aqfs_snippet/test.json \
    --nli_model_path alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli \
    --claimsplit_model_path {path/to/the/fine-tuned/claimsplit/model/checkpoint} \
    --nli_batch_size 256 \
    --claimsplit_batch_size 512 
``` 

如果遇到OOM错误，尝试减小`nli_batch_size`和`claimsplit_batch_size`。

## 进行AQFS任务的实验

### 有监督微调
我们使用[deepspeed library](https://github.com/microsoft/DeepSpeed)进行LLM微调，相关的配置文件是`src/deepspeed.json`。
下面是在这个任务上微调`THUDM/chatglm3`模型的示例。
```bash
cd src
deepspeed --num_gpus=8 aqfs/main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --data_dir ../data/aqfs_snippet \
    --source_column prompt \
    --target_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm3-6b \
    --output_dir {path/to/output/directory} \
    --overwrite_output_dir \
    --max_source_length 1280 \
    --max_target_length 400 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --logging_steps 2 \
    --learning_rate 2e-5 \
    --fp16
```

我们在8张A100-40GB的GPU上测试了上述命令。然而，通过启用CPU offloading，可能在更少的计算资源上运行代码。可以参阅[此教程](https://huggingface.co/docs/transformers/v4.38.2/deepspeed)获取更多信息。

### 推理
下面是使用微调模型进行推理的示例。
```bash
cd src
NUM_GPUs=8
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUs aqfs/main.py \
    --do_predict \
    --data_dir ../data/aqfs_snippet \
    --source_column prompt \
    --target_column summary \
    --overwrite_cache \
    --model_name_or_path {path/to/the/model/checkpoint} \
    --output_dir {path/to/output/directory} \
    --overwrite_output_dir \
    --max_source_length 1280 \
    --max_target_length 400 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate 
```

上述命令执行了数据并行推理，其中每个GPU会加载模型的全部权重并处理一个小批量的数据。我们还提供了用于模型并行推理的脚本，其中模型的权重被切分到多个GPU上。如果在运行上述命令时遇到OOM错误，请尝试以下命令来使用模型并行推理。

```bash
python aqfs/inference.py --{same arguments as above}
```

#### 少样本提示
下面是使用少样本提示（fewshot prompting）进行推理的示例。

```bash
cd src
NUM_GPUs=8
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUs aqfs/main.py \
    --do_predict \
    --use_chat_format \
    --data_dir ../data/aqfs_snippet \
    --source_column prompt \
    --target_column summary \
    --exemplar_id a18f3958-3d98-4e1a-bca6-d8c83712ec64 \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm3-6b \
    --output_dir {path/to/output/directory} \
    --overwrite_output_dir \
    --max_source_length 3680 \
    --max_target_length 400 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate 
```

其中有两个额外参数`--use_chat_format`和`--exemplar_id`。同样可以使用`aqfs/inference.py`进行模型并行推理。


#### 长文本设置

在长文本设置中，模型基于完整的网页内容来进行摘要。为了采用这种设置，需要将`--data_dir`设置为`../data/aqfs_full_chunk256`或`../data/aqfs_full_chunk512`，并将`--max_source_length`设置为7680。

### 结果评估
我们使用先前准备的评估器来进行自动评估。下面是对模型输出进行评估的示例。

```bash
cd src
python aqfs/eval.py \
    --pred_file {path/to/model/outputs} \
    --nli_model_path alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli \
    --claimsplit_model_path {path/to/the/fine-tuned/claimsplit/model/checkpoint} \
    --nli_batch_size 256 \
    --claimsplit_batch_size 512 
```
如果遇到OOM错误，尝试减小`nli_batch_size`和`claimsplit_batch_size`。

### ChatGPT 输出结果

运行下面的命令来下载GPT-3.5和GPT-4的输出结果。
```bash
wget https://huggingface.co/datasets/HarlynDN/WebCiteS/resolve/main/gpt_outputs.tar
tar -xvf gpt_outputs.tar
echo "deleting tar file"
rm gpt_outputs.tar
```
