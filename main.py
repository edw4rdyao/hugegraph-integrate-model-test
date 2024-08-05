from tests.dgi_test import dgi_test
from tests.grace_test import grace_test
from tests.jknet_test import jknet_test

# dgi test
dgi_test(dataset_name="cora", self_loop=True)
dgi_test(dataset_name="citeseer")
dgi_test(dataset_name="pubmed")

# grace test
grace_test(dataset_name="cora", lr=5e-4, n_hidden=128, n_out_feats=128, act_fn="relu", temp=0.4, n_epochs=200)
grace_test(dataset_name="citeseer", lr=1e-3, n_hidden=256, n_out_feats=256, act_fn="prelu", temp=0.9, n_epochs=200)
grace_test(dataset_name="pubmed", lr=1e-3, n_hidden=256, n_out_feats=256, act_fn="relu", temp=0.7, n_epochs=1500)

# jknet test
jknet_test(dataset_name="cora", n_layers=6, mode="max")
jknet_test(dataset_name="cora", n_layers=6, mode="cat")
jknet_test(dataset_name="cora", n_layers=6, mode="lstm")
jknet_test(dataset_name="citeseer", n_layers=6, mode="max")
jknet_test(dataset_name="citeseer", n_layers=6, mode="cat")
jknet_test(dataset_name="citeseer", n_layers=6, mode="lstm")
