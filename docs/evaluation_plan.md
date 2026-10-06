# Evaluation Plan

The evaluation script runs representative AI/ML search queries against Pinecone and records latency plus retrieval proxy metrics.

## Metrics

- Query latency per query.
- Mean, median, and P95 latency.
- Category hit rate at k when expected categories are configured.
- Precision@k only when manually reviewed relevant IDs are supplied.
- Metadata-filter correctness for category and year constraints.

## Important Limitation

Category hit rate and keyword/concept checks are proxy metrics. They do not prove a paper is relevant. Human relevance judgments can be added by filling `evaluation/manual_evaluation_template.csv` and copying confirmed relevant record IDs into `evaluation/queries.json`.

## Outputs

- `evaluation/results.csv`
- `evaluation/summary.md`

