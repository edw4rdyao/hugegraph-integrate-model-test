# hugegraph-integrate-model-test

## Run

```commandline
python main.py
```

## Model test result

### DGI

|    Dataset    | Cora | Citeseer | Pubmed |
|:-------------:|:----:|:--------:|:------:|
|     paper     | 82.3 |   71.8   |  76.8  |
|     this      | 81.3 |   70.4   |  77.3  |

### GRACE

|    Dataset    | Cora | Citeseer | Pubmed |
|:-------------:|:----:|:--------:|:------:|
|     paper     | 83.3 |   72.1   |  86.7  |
|     this      | 82.8 |   72.5   |  86.2  |

### JKNet

- Cora

|   Cora   | max  | cat  | lstm |
|:--------:|:----:|------|------|
|  paper   | 89.6 | 89.1 | 85.8 |
|   this   | 85.1 | 86.7 | 86.4 |

| Citeseer | max  | cat  | lstm |
|:--------:|:----:|------|------|
|  paper   | 77.7 | 78.3 | 74.7 |
|   this   | 60.8 | 74.6 | 73.2 |

