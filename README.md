# Dissecting LLM-Based Program Repair

An empirical study of how model capability, repair strategy, and fault type affect LLM-based automated program repair.

## Prerequisites

### Environment Setup

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n coderepair python=3.10 -y
conda activate coderepair

# PyTorch (GPU)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Transformers
conda install transformers==4.36.2
conda install accelerate==0.23.0
pip install evaluate==0.4.3

# Data & ML
pip install numpy==1.26.0 scikit-learn==1.3.0 pandas==2.1.0
pip install imbalanced-learn==0.11.0
pip install gensim==4.3.0

# OpenAI SDK (for GPT-based generation)
pip install openai==1.6.0

# CodeBLEU
pip install codebleu==0.7.0
pip install tree-sitter-java==0.21

# Other
pip install fire tqdm
```

### Java 8 + Jasper

```bash
conda install openjdk=8
cd jasper
mkdir -p target
javac -cp ".:lib/*" -d target src/main/java/clm/jasper/*.java
cd ..
```

### Benchmark Data

```bash
mkdir -p datasets

# QuixBugs
git clone https://github.com/jkoppel/QuixBugs.git datasets/quixbugs
cp -r datasets/quixbugs/java_programs datasets/quixbugs/java_programs_bak

# Defects4J
git clone https://github.com/rjust/defects4j.git datasets/defects4j
cd datasets/defects4j
git checkout tags/v2.0.1 -b d4j-2.0.1 --force
cd ../..
# Install Defects4J
cd datasets/defects4j
cpan App::cpanminus
./init.sh
export PATH=$PATH:$(pwd)/framework/bin
cd ../..

# HumanEval-Java
cd datasets
wget https://raw.githubusercontent.com/lin-tan/clm/refs/heads/main/humaneval-java/humaneval-java.tar.gz
tar -xzvf humaneval-java.tar.gz
cp -r humaneval-java/src humaneval-java/src_bak
cd ..

# Create project backup dirs (used by validation)
mkdir -p datasets/quixbugs/proj
mkdir -p datasets/humaneval-java/proj
mkdir -p datasets/defects4j/proj
```

### Models

For local inference, place the required model checkpoints under `models/`.
GPT-based models are accessed through the OpenAI API.

| Model | Path | Used For |
|-------|------|----------|
| GPT-4o (API) | `$OPENAI_API_KEY` | All strategies |
| CodeLlama-Instruct-7b | `models/CodeLlama-Instruct-7b` | Local inference |
| DeepSeek-Coder-6.7b | `models/deepseekcoder-6.7b` | Local inference |
| DeepSeek-Coder-1.3b | `models/deepseekcoder-1.3b` | Local inference |
| UnixCoder-base | `models/unixcoder-base` | RAG / COT retrieval |

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"      # default
export JASPER_LIB_DIR="./jasper/lib"                     # optional, defaults to ./jasper/lib
```

## Run the Pipeline

Edit `src/run_all_scripts.sh` and set the following variables at the top:

```bash
model_type="gpt-4o"                   # gpt-4o, CodeLlama-Instruct-7b, deepseekcoder-6.7b
enhance_methods="zero-shot"           # zero-shot, few-shot, rag, cot
use_bug_type="false"                  # set to "true" to include bug type in the prompt
benchmark_names=("quixbugs_c1" "humaneval-java_c1" "defects4j_c1")   # c1 by default
```

- **c1 vs c2**: Use `c1` by default. Switch to `c2` only when studying the effect of buggy-line context.
- **Bug type**: Set `use_bug_type="true"` to enable the bug-type-aware prompt (`repair_prompt_open_source_type.txt`).
This option is only required when reproducing the bug-type-aware experiments.


```bash
bash src/run_all_scripts.sh
```

## Pipeline Overview

```
benchmark data ──→ patch generation ──→ patch validation ──→ analysis
   (JSON)         (generate_patch.py      (patch_validation.py    (analyse_benchmark.py)
                   or rag_cot/rag.py)       + per-benchmark
                                            validators)
```

### Step 1: Patch Generation

- **Zero-shot / Few-shot**: `src/inference_and_validation/generate_patch.py`
- **RAG / COT**: `src/inference_and_validation/rag.py`

Supports: `zero-shot`, `few-shot`, `rag`, `cot`

### Step 2: Patch Validation

`src/inference_and_validation/patch_validation.py` dispatches to:
- `defects4j_patch_validate.py`
- `humaneval_patch_validate.py`
- `quixbugs_patch_validate.py`

### Step 3: Analysis

`src/analysers/analyse_benchmark.py` computes repair rates per bug category.

## Project Structure

```
Dissecting-LLM-APR/
├── datasets/
│   ├── benchmarks/           # Benchmark input data (JSON)
│   │   ├── quixbugs_c1.json
│   │   ├── quixbugs_c2.json
│   │   ├── humaneval-java_c1.json
│   │   ├── humaneval-java_c2.json
│   │   ├── defects4j_c1.json
│   │   └── defects4j_c2.json
│   ├── prompt/                   # Prompt templates
│   │   ├── repair_prompt_open_source.txt
│   │   ├── repair_prompt_open_source_type.txt
│   │   ├── repair_prompt_open_source_rag.txt
│   │   └── repair_prompt_cot.txt
│   ├── rag/                      # RAG cache & knowledge base
│   │   ├── Datasets/             # Retrieval knowledge base
│   │   │   ├── defects4j.json
│   │   │   ├── humaneval.json
│   │   │   └── quixbugs.json
│   │   └── {model}/              # TF-IDF cache per model
│   ├── quixbugs/                 # QuixBugs Java test suites
│   ├── humaneval-java/           # HumanEval-Java test suites
│   ├── defects4j/                # Defects4J checkouts
│   └── results/                  # Generated outputs + validation results
├── src/
│   ├── run_all_scripts.sh        # One-click pipeline
│   ├── analysers/
│   │   └── analyse_benchmark.py
│   └── inference_and_validation/
│       ├── generate_patch.py     # Zero-shot / Few-shot generation
│       ├── rag.py                # RAG / COT generation
│       ├── patch_validation.py   # Validation dispatcher
│       ├── apr_utils.py          # Shared utilities
│       ├── validation_utils.py   # Test output parsing
│       ├── result_look.py        # pass@k computation
│       ├── defects4j_patch_validate.py
│       ├── humaneval_patch_validate.py
│       └── quixbugs_patch_validate.py
├── models/                       # Model checkpoints (not tracked)
└── jasper/                       # Jasper Java parser library
```


## Supported Benchmarks

| Benchmark | c1 (original) | c2 (expanded context) |
|-----------|:---:|:---:|
| QuixBugs (Java) | `quixbugs_c1` | `quixbugs_c2` |
| HumanEval-Java | `humaneval-java_c1` | `humaneval-java_c2` |
| Defects4J | `defects4j_c1` | `defects4j_c2` |


## Acknowledgements

This repository incorporates and extends implementations from several prior open-source projects. We sincerely thank the original authors for releasing their code.

Specifically:

- The implementation of the retrieval-based repair pipeline is adapted from **ThinkRepair: Self-Directed Automated Program Repair**.
- The implementation of the patch generation pipeline is partially adapted from **Exploring Parameter-Efficient Fine-Tuning of Large Language Models on Automated Program Repair**.
- The project organization and patch validation framework are adapted from **The Impact of Fine-tuning Large Language Models on Automated Program Repair**.

These implementations have been modified and extended to support the experimental settings used in this study.