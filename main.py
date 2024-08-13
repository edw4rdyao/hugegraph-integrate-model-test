from tests.dgi_test import dgi_test
from tests.evolve_gcn_test import evolve_gcn_test
from tests.grace_test import grace_test
from tests.grand_test import grand_test
from tests.han_test import han_test
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

# grand test
grand_test(dataset_name="cora", order=8, sample=4, lam=1.0, temp=0.5, p_drop_input=0.5, p_drop_hidden=0.5,
           lr=1e-2, early_stopping=100)

grand_test(dataset_name="citeseer", order=2, sample=2, lam=0.7, temp=0.3, p_drop_input=0.0, p_drop_hidden=0.2,
           lr=1e-2, early_stopping=100)

grand_test(dataset_name="pubmed", order=5, sample=4, lam=1.0, temp=0.2, p_drop_input=0.6, p_drop_hidden=0.8,
           lr=0.2, bn=True)

# han test
han_test(dataset_name="acm", n_hidden=4, p_drop=0.6, n_heads=[8],
         lr=0.005, wd=0.001, n_epochs=200, seed=42)

# evolve gcn test
evolve_gcn_test(dataset_name="elliptic")
