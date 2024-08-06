# hugegraph-integrate-model-test

## Run

```commandline
python main.py
```

## Model test result

### DGI

|    Dataset    |    Cora    |  Citeseer  |   Pubmed   |
|:-------------:|:----------:|:----------:|:----------:|
|     paper     | 82.3(±0.6) | 71.8(±0.7) | 76.8(±0.6) |
|     this      |    81.3    |    70.4    |    77.3    |

### GRACE

| Dataset |    Cora    |  Citeseer  |   Pubmed   |
|:-------:|:----------:|:----------:|:----------:|
|  paper  | 83.3(±0.4) | 72.1(±0.5) | 86.7(±0.1) |
|  this   |    82.8    |    72.5    |    86.2    |

### JKNet

- Cora

|   Cora   |    max     | cat        | lstm       |
|:--------:|:----------:|------------|------------|
|  paper   | 89.6(±0.5) | 89.1(±1.1) | 85.8(±1.0) |
|   this   |    85.1    | 86.7       | 86.4       |

- Citeseer

| Citeseer |    max     | cat        | lstm       |
|:--------:|:----------:|------------|------------|
|  paper   | 77.7(±0.5) | 78.3(±0.8) | 74.7(±0.9) |
|   this   |    70.8    | 74.6       | 73.2       |

### GRAND

| Dataset |    Cora    |  Citeseer  |   Pubmed   |
|:-------:|:----------:|:----------:|:----------:|
|  paper  | 85.4(±0.4) | 75.4(±0.4) | 82.7(±0.6) |
|  this   |    84.6    |    75.4    |    82.3    |

