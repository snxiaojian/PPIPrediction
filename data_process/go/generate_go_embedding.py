import anc2vec.train as builder
import numpy as np

obo_file = "data/go/go.obo"
es = builder.fit(obo_file, embedding_sz=512, batch_sz=128, num_epochs=100)


with open("data/go/go_embedding.npy", "wb") as f:
    np.save(f, es)