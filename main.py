from tests.dgi_test import dgi_test
from tests.grace_test import grace_test

# dgi test
dgi_test_args = [
    [
        '--dataset', 'cora',
        '--gpu', '-1',
        '--self-loop'
    ],
    [
        '--dataset', 'citeseer',
        '--gpu', '-1'
    ],
    [
        '--dataset', 'pubmed',
        '--gpu', '-1'
    ]
]
for arg in dgi_test_args:
    dgi_test(arg)

# grace test
grace_test_args = [
    [
        "--dataname", "cora",
        "--epochs", "200",
        "--lr", "5e-4",
        "--wd", "1e-5",
        "--hid_dim", "128",
        "--out_dim", "128",
        "--act_fn", "relu",
        "--der1", "0.2",
        "--der2", "0.4",
        "--dfr1", "0.3",
        "--dfr2", "0.4",
        "--temp", "0.4",
        "--gpu", "-1"
    ],
    [
        "--dataname", "citeseer",
        "--epochs", "200",
        "--lr", "1e-3",
        "--wd", "1e-5",
        "--hid_dim", "256",
        "--out_dim", "256",
        "--act_fn", "prelu",
        "--der1", "0.2",
        "--der2", "0.0",
        "--dfr1", "0.3",
        "--dfr2", "0.2",
        "--temp", "0.9",
        "--gpu", "-1"
    ],
    [
        "--dataname", "pubmed",
        "--epochs", "1500",
        "--lr", "1e-3",
        "--wd", "1e-5",
        "--hid_dim", "256",
        "--out_dim", "256",
        "--act_fn", "relu",
        "--der1", "0.4",
        "--der2", "0.1",
        "--dfr1", "0.0",
        "--dfr2", "0.2",
        "--temp", "0.7",
        "--gpu", "-1"
    ]
]

for arg in grace_test_args:
    grace_test(arg)
