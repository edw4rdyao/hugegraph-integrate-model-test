from tests.dgi_test import dgi_test
from tests.grace_test import grace_test

# dgi test
for d in ["cora", "citeseer", "pubmed"]:
    dgi_test(dataset_name=d)

# grace test
grace_test(dataset_name="cora", lr=5e-4, n_hidden=128, n_out_feats=128, n_layers=2, act_fn="relu",
           der1=0.2, der2=0.4, dfr1=0.3, dfr2=0.4, temp=0.4, epochs=200, wd=1e-5)
grace_test(dataset_name="citeseer", lr=1e-3, n_hidden=256, n_out_feats=256, n_layers=2, act_fn="prelu",
           der1=0.2, der2=0.0, dfr1=0.3, dfr2=0.2, temp=0.9, epochs=200, wd=1e-5)
grace_test(dataset_name="pubmed", lr=1e-3, n_hidden=256, n_out_feats=256, n_layers=2, act_fn="relu",
           der1=0.4, der2=0.1, dfr1=0.0, dfr2=0.2, temp=0.7, epochs=1500, wd=1e-5)
