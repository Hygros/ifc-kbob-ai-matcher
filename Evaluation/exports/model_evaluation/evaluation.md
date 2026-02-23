# Evaluation

## Summary

| Model | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | MAP@10 | Recall@10 | Cov@95 | AutoCov | AutoAcc | Manual Hit@10 | AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kforth/IfcElement2ConstructionSets | 0.500 | 0.500 | 1.000 | 0.583 | 0.678 | 0.583 | 1.000 | 0.500 | 0.000 | 0.000 | 1.000 | 0.125 |
| google/embeddinggemma-300m | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| BAAI/bge-m3 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| intfloat/multilingual-e5-large | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| intfloat/multilingual-e5-base | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| sentence-transformers/LaBSE | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.000 | 0.125 |
| kforth/IfcMaterial2MP | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| google-bert/bert-base-german-cased | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| google-bert/bert-base-multilingual-uncased | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |
| google-bert/bert-base-multilingual-cased | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 | 0.125 |

## Coverage targets

Coverage mit Accuracy-Bedingung auf Testset:

- `coverage_at_90acc`, `coverage_at_95acc`, `coverage_at_97acc`, `coverage_at_99acc`
