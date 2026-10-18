# RAGTruth-FMN

Concept- and link-level hallucination labels for all 1,350 RAGTruth test answers of Mistral-7B-Instruct, Llama-2-7B-chat and Llama-2-13B-chat (150 answers per task: QA, Summary, Data2txt), with the score of every method in the paper.

An element is labelled hallucinated when any of its occurrences overlaps a human-annotated RAGTruth hallucination span, and it inherits that span's type. A link counts as covered when either of its words lies in the span. `stats.md` gives counts per model and task.

## ragtruth_fmn_elements.csv.gz (one row per concept or link, 273,456 rows)

| Column | Meaning |
|---|---|
| model, task, response_id | generator, RAGTruth task, RAGTruth response identifier |
| element_type | `node` (concept) or `edge` (link) |
| element | the lemma, or the two lemmas of a link |
| occurrences | number of times the element occurs in the answer |
| degree | degree of the node in the answer's network |
| hallucinated | 1 if the element overlaps an annotated hallucination span |
| hallucination_types | RAGTruth types of the overlapping spans (Evident/Subtle Conflict, Evident/Subtle Baseless Info) |
| score_zfmn_full_P, _d, _u | **Z-FMN**: fused opinion (projected probability, disbelief, uncertainty); risk = 1 - P |
| score_zfmn_model_P | model opinion alone (projected probability) |
| score_zfmn_source_P, _u | source (entailment) opinion: projected probability and uncertainty |
| score_zfmn_context_P, _u | context-conditioned model opinion (ablation in the paper) |
| score_nli_entail, _neutral, _contra | DeBERTa-large-MNLI probabilities of the element's sentence against the source |
| score_selfcheck_nli | SelfCheckGPT-NLI score of the element's sentence (higher = more likely hallucinated) |
| score_mean_logprob | mean token log-probability of the element |
| emo_valence, emo_n_emotions | EmoAtlas valence and number of NRC emotions of the concept (concepts only) |
| emo_frame_valence, emo_frame_valence_spread | mean and spread of the valence of the concept's neighbours |

## ragtruth_fmn_answers.csv.gz (one row per answer, 1,350 rows)

`hallucinated` and `types` give the answer label. Answer-level scores (higher = more likely hallucinated): `resp_edge_tri_fused_P_max` is Z-FMN, the risk 1 - P of the answer's riskiest link; `resp_selfcheck_nli`, `resp_nli_not_entailed` and `resp_neg_mean_logprob` are the baselines. The other `resp_{node,edge}_*_{max,mean}` columns are the maximum and mean element risk of other opinions.
