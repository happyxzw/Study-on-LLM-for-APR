# Inference & Validation

## Architecture

```
run_all_scripts.sh
├── generate_patch.py              # zero-shot / few-shot
│   ├── apr_utils.py               # OpenAI client, model loader, prompt formatter
│   └── datasets/prompt/*.txt      # prompt templates
│
├── rag.py                         # RAG / COT
│   ├── apr_utils.py               # (shared)
│   ├── datasets/prompt/*.txt
│   └── datasets/rag/Datasets/    # knowledge base for retrieval
│       ├── defects4j.json
│       ├── humaneval.json
│       └── quixbugs.json
│
├── patch_validation.py            # validation dispatcher
│   ├── defects4j_patch_validate.py
│   ├── humaneval_patch_validate.py
│   ├── quixbugs_patch_validate.py
│   ├── validation_utils.py
│   └── result_look.py
│
└── ../analysers/analyse_benchmark.py  # repair rate analysis
```

## Files

| File | Description |
|------|-------------|
| `generate_patch.py` | Zero-shot / few-shot patch generation |
| `rag.py` | RAG / COT patch generation with TF-IDF retrieval |
| `patch_validation.py` | Dispatches to benchmark-specific test runners |
| `apr_utils.py` | Shared: `create_model_and_tokenizer`, `prompt_text`, OpenAI `client` |
| `prompter.py` | `Prompter` class (unused by current pipeline) |
| `validation_utils.py` | Test output parsing, log writing |
| `result_look.py` | pass@k computation |
| `defects4j_patch_validate.py` | Defects4J test runner |
| `humaneval_patch_validate.py` | HumanEval-Java test runner |
| `quixbugs_patch_validate.py` | QuixBugs test runner |
| `models.py` | Model class definitions (unused by current pipeline) |
