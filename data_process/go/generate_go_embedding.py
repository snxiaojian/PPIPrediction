import anc2vec.train as builder
import numpy as np
import sys
sys.path.append("./")
from ppi_network.static_args import *


obo_file = "data/go/go.obo"
es = builder.fit(obo_file, embedding_sz=residue_embedding_size, batch_sz=128, num_epochs=100)


with open("data/go/go_embedding.npy", "wb") as f:
    np.save(f, es)