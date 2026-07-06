# Replication Package for _"Dissecting LLM-Based Program Repair: An Empirical Study of Model Capability, RepairStrategy, and Fault Type"_
## Prerequisites

### Setting up the environment and installing dependencies

The Python requirements and the environment can be set up as follows:

```bash
# Add the Conda-forge to the channels
conda config --add channels conda-forge
conda config --set channel_priority strict
	
# Set up the environment
conda create --name coderepair python=3.11.5
conda activate coderepair
	
# Install main ML requirements
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers==4.36.2
conda install accelerate==0.23.0
pip install peft==0.6.0
pip install evaluate==0.4.3
pip install numpy==1.26.0
pip install wandb==0.15.12

# CodeBleu
pip install codebleu==0.7.0
pip install tree-sitter-java==0.21

# other requirements
pip install zenodo-get tqdm click
```

Next, we need to install [Jasper](https://github.com/lin-tan/clm/tree/main/jasper): an AST-based Java Parser for Program Repair. A copy of the code is provided, along with the authors' instructions, in `jasper/README.md`.  Note that Java 8 is needed for compiling Jasper

```bash
conda install openjdk=8
cd jasper
mkdir target
javac -cp ".:lib/*" -d target src/main/java/clm/jasper/*.java src/main/java/clm/codet5/*.java src/main/java/clm/codegen/*.java src/main/java/clm/plbart/*.java src/main/java/clm/incoder/*.java src/main/java/clm/finetuning/*.java
```

### Setting up APR benchmarks and fine-tuning datasets

The following APR benchmarks and fine-tuning datasets need to be installed:
[Quixbugs](https://github.com/jkoppel/QuixBugs),
[Defects4j](https://github.com/rjust/defects4j), and 
[HumanEval-Java](https://github.com/lin-tan/clm/tree/main/humaneval-java), and
[CLM](https://zenodo.org/records/7559208).
   
You can download them from within the project root, using:

```bash
mkdir -p datasets

# Download QuixBugs, Defects4J, and CLM
git clone https://github.com/jkoppel/QuixBugs.git datasets/QuixBugs
git clone https://github.com/rjust/defects4j.git datasets/defects4j
zenodo_get 7559208 -o ./datasets/clm
mv QuixBugs quixbugs

# Humaneval-Java
cd datasets
wget https://raw.githubusercontent.com/lin-tan/clm/refs/heads/main/humaneval-java/humaneval-java.tar.gz
tar -xzvf humaneval-java.tar.gz
cd humaneval-java
cp src src_bak
cd ..

# Quixbugs
cd quixbugs
cp -r java_programs java_programs_bak
cd ..

# Set the correct version for Defects4J
cd defects4j
git checkout tags/v2.0.1 -b d4j-2.0.1 --force
cd ..

# Add bug location info
mv quixbugs_loc.txt quixbugs
mv humaneval_loc.txt humaneval-java
mv defects4j_loc.txt defects4j

# Create project folders for the benchmarks
mkdir datasets/humaneval-java/proj
mkdir datasets/quixbugs/proj
mkdir datasets/defects4j/proj

# Make a temporary folder to store copies of the benchmarks (so they can easily be reinitialized)
cd ..
mkdir tmp
cp -r humaneval-java tmp
cp -r quixbugs tmp
cp -r defects4j tmp

# Install Defects4j
cd defects4j
conda install perl
conda install compilers
conda install conda-forge gcc_linux-64 sysroot_linux-64=2.17
cpan App::cpanminus
./init.sh
export PATH=$PATH:$(pwd)/framework/bin
```

### Environment.yml

For reference, we provide a file `environment.yml` in the root of the replication package that includes our complete conda environment after following these installation steps. 
# Exploring Parameter-Efficient Fine-Tuning of Large Language Model on Automated Program Repair

## Dependency

### Python

* Python 3.9.17
* PyTorch 2.0.1
* Huggingface transformers 4.35.2
* wandb
* pef 0.6.2

- accelerate 0.24.1

- datasets 2.13.0

- trl

- fire

* nvitop

### Others

- Java 8

## About PEFT weights  

- We have released 4 PEFT weights of each base model on HuggingFace, trained on APR-INSTRUCTION dataset.
- [PEFT Weights here](https://huggingface.co/survoltli/llmpeft4apr)


## Content
The file structure of the artifact is as follow:

cd /home/chenshiping/llmpeft4apr-main/train_scripts
sh lora_16_train_apr.sh
bash lora_16_apr_validation.sh
### APR-INSTRUCTION_construct;

- contains source code of constructing `APR-INSTRUCTION` ,base existing APR dataset[1]

### **codellama_7b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ and Full-model Fine-tuning

- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `codellama-7b-hf` and `codellama-7b-hf` with peft weights, validation results of generated pacthes

### **codellama_13b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `codellama-13b-hf` and `codellama-13b-hf` with peft weights, validation results of generated pacthes

### **deepseek_coder_6.7b:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `Deepseek-Coder Base 6.7B` and `Deepseek-Coder Base 6.7B` with peft weights, validation results of generated pacthes

### **llama2_7b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `Llama-2-7b-hf` and `Llama-2-7b-hf` with peft weights, validation results of generated pacthes

### instruction_tuning_dataset

- Instruction Dataset used this paper
  - apr_instruction_30k.json: the APR instruction dataset constructed this paper
  - oss_instrcution_30k.json: 30k random selection of OSS-Instruction Dataset
  - code_alpaca_20k.json: Code Alpaca Instruction Dataset
  - The rest of data is used for RQ3 to explore the impact of training data size for performance, which is parted as 10k, 15k, 20k and 25k

### **inference_and_validation_src:**

- This directory consists of source code used for patches generation and validation of LLMs
    |  file name  |       description     |
    |  :----:             |       :----:          |
    | defects4j_patch_validate.py | patches generation and validation on Defects4j benchmark |
    | humaneval_patch_validate.py | patches generation and validation on Humaneval-Java benchmark |
    | quixbugs_patch_validate.py | patches generation and validation on Quixbugs benchmark |
    | peft_patch_validation.py | Entry of model validation with PEFT methods, and then select different scripts for verification |
    | fmft_generate_patch.py | Entry of model validation with Full-model fine-tuning, and then select different scripts for verification |
    | generate_patch_infill.py | Entry of CodeLlama 7b validation with no fine-tuning and infill templates, and then select different scripts for verification |
    | prompter.py | convert instances of benchmark to instruction |
    | result_look.py | record $pass@k$ of each validation |

### **inference_scripts:**

- This directory consists of bash scripts used for patches generation and validation of LLMs


- each script is formed as `model name`\_`Fine-tuning method`\_`instruction dataset of Fine-tuning`\_validation.sh

### **train_scripts:**

- This directory consists of bash scripts used for LLMs training 
- each script is formed as `model name`\_instrcution\_`Fine-tuning method`\_`hyper-parameters(Optional)`\_train\_`instruction dataset of Fine-tuning`\_validation.sh

### train_src:


- This directory consists of source code used for LLM trainnin

    |  file name    |       description     |
    |  :----:             |       :----:          |
    | sfttrain_peft.py | Training code for PEFT methods |
    | sfttrain_ft.py | Training code for Full-model Fine-tuning |
    |   prompter.py    |  Add additional prompt for instruction   |
    

### results_hyper_parameters:

- This directory consists of results of patches generation and validation in experiments of RQ3




## NOTICE  

- Due to the size of `Fine-tuning weights`  is too large, so we do not upload them on Github now
- Considering the anonymous review,  we will  release weights after review


## Cites  

```
[1] Zhu, Qihao, et al. "A syntax-guided edit decoder for neural program repair." Proceedings of the 29th ACM joint meeting on European software engineering conference and symposium on the foundations of software engineering. 2021.
```


