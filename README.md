# Z-FMN: hallucination detection on the concepts and links of an answer

Code and data for the paper *Hallucination detection in retrieval-augmented language models by Z-number evidence fusion on concept networks* (Sweta Kaman, under review).

Z-FMN turns each answer of a retrieval-augmented language model into a textual forma mentis network: concepts are nodes, and two concepts are linked when they lie within three dependency hops. Every concept and link gets a Z-number whose reliability is a subjective-logic opinion (belief, disbelief, uncertainty). Three evidence sources are fused cumulatively, so each counts in proportion to how much evidence it carries:

1. **Model opinion**: the generator's token confidence, pooled over the answer and 10 resampled answers.
2. **Source opinion**: entailment of the element's sentence against the retrieved source (DeBERTa-large-MNLI). A neutral verdict adds uncertainty, not disbelief.
3. **Consistency opinion**: the SelfCheckGPT-NLI score against the resamples.

The hallucination risk of an element is `1 - P`, where `P = b + u/2`. Nothing is trained.

## Main results (RAGTruth test set, 1,350 answers, Mistral-7B, Llama-2-7B, Llama-2-13B)

| Method | Link AUROC | Concept AUROC |
|---|---|---|
| **Z-FMN** | **0.793** | 0.762 |
| Source NLI | 0.767 | 0.753 |
| SelfCheckGPT-NLI | 0.725 | 0.729 |
| Token log-probability | 0.610 | 0.600 |
| Supervised stacking (uses labels) | 0.801 | 0.786 |

A reviewer who checks the riskiest 20% of links finds 59.7% of the hallucinated links with Z-FMN, against 49.6% with SelfCheckGPT-NLI. All numbers are means over the nine model x task datasets; the paper gives 95% bootstrap intervals.

## Repository layout

```
zfmn/            the method: network construction, evidence, subjective logic, Z-numbers, metrics
run_pilot.py     the pipeline: generate + score, NLI and SelfCheckGPT, evaluate (one model per run)
scripts/         analyses that produce every table and figure in the paper
notebooks/       zfmn_full_study.ipynb, the Kaggle notebook used for the full study
data/            RAGTruth-FMN benchmark (concept- and link-level labels and all scores)
results/         the final tables and numbers reported in the paper
figures/         the paper's figures (PDF)
tests/           unit tests (no GPU needed)
```

## Install

Python 3.10 or later.

```bash
pip install -r requirements.txt
pip install --no-deps emoatlas          # emotion lexicons, used for the emotion analysis only
python -m spacy download en_core_web_sm
python tests/test_core.py               # prints five "ok" lines
```

## Use the benchmark without a GPU

`data/ragtruth_fmn_elements.csv.gz` has one row per concept or link (273,456 rows) and `data/ragtruth_fmn_answers.csv.gz` one row per answer (1,350 rows). See [data/README.md](data/README.md) for every column. For example, to reproduce the review-budget analysis:

```bash
python scripts/review_budget.py data/ragtruth_fmn_elements.csv.gz results/review_budget.json
```

## Reproduce the full study

The full study needs a GPU. It was run on Kaggle's free tier (two T4 GPUs, 4-bit models), about 60 GPU-hours for the three models. The simplest route is to upload `notebooks/zfmn_full_study.ipynb` to Kaggle, add an `HF_TOKEN` secret with access to the Llama-2 and Mistral models, and run all cells. The notebook resumes where it stopped if a session times out.

From the command line, for each model:

```bash
COMMON="--split test --tasks QA,Summary,Data2txt --n-samples 10 --load-in-4bit"
python run_pilot.py --model-label mistral-7B-instruct $COMMON --stage generate --out runs/full_mistral-7B-instruct
python run_pilot.py --model-label mistral-7B-instruct $COMMON --stage nli      --out runs/full_mistral-7B-instruct
python run_pilot.py --model-label mistral-7B-instruct $COMMON --stage evaluate --out runs/full_mistral-7B-instruct
```

Model labels: `mistral-7B-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat`. RAGTruth is downloaded automatically.

Then, with `RUNS="Llama-2-7B=runs/full_llama-2-7b-chat Mistral-7B=runs/full_mistral-7B-instruct Llama-2-13B=runs/full_llama-2-13b-chat"`:

```bash
python scripts/z_ablation.py $RUNS > results/z_ablation_final.md                 # main comparison, ablations, calibration
python scripts/full_study_analysis.py $RUNS > results/full_study_final_tables.md  # per-dataset tables
python scripts/rq4_emotion_analysis.py $RUNS > results/rq4_final.md               # emotion models
python scripts/extra_analysis.py results/extra_final.json $RUNS                   # types, scale, sensitivity, answer level
python scripts/build_benchmark.py data $RUNS                                      # the benchmark files
python scripts/make_paper_figures.py figures results/extra_final.json results/rq4_final.md $RUNS
python scripts/make_eaai_figures.py figures data/ragtruth_fmn_elements.csv.gz
```

The raw run folders (generations, resamples, NLI outputs, about 450 MB) are archived separately; see the Data availability section of the paper.

## Licence

Code: MIT (see `LICENSE`). The benchmark is derived from RAGTruth (MIT licence, Niu et al., 2024); it contains labels, scores and RAGTruth response identifiers, not the source passages, which join back to RAGTruth by `response_id`.

## Citation

If you use the code or the benchmark, please cite the paper (see `CITATION.cff`) and RAGTruth:

> Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., Zhang, T., 2024. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. ACL 2024.
