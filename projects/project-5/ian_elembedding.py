import mowl
mowl.init_jvm("4g")

from pathlib import Path
import numpy as np, torch
from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings

torch.manual_seed(42); np.random.seed(42)

DATA = Path(__file__).parent.parent.parent / "documentation" / "mowl-demo"
dataset = PathDataset(str(DATA / "toy_train.ttl"),
                      validation_path=str(DATA / "toy_valid.ttl"),
                      testing_path=str(DATA / "toy_test.ttl"))

# mowl 2.x on Windows crashes writing its normalization cache (backslash path),
# and only caches when it knows the file paths, so clear them after loading.
dataset.ontology_path = dataset.validation_path = dataset.testing_path = None


EPOCHS = 50
model = ELEmbeddings(dataset, embed_dim=16, learning_rate=0.01)

print("Training datasets:")
for name, ds in model.training_datasets.items():
    print(f"  {name}: {len(ds)}")


model.eval_gci_name = "gci0"   # validate on plain subClassOf axioms
model.train(epochs=EPOCHS)


emb = model.class_embeddings
def cos(a, b):
    va, vb = emb[f"http://example.org#{a}"], emb[f"http://example.org#{b}"]
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

print("\nSimilarity scores:")
for a, b in [("Dog", "Mammal"), ("Mammal", "Animal"), ("Dog", "Animal")]:
    print(f"  {a}-{b}: {cos(a, b):.2f}")
print("\n✅ Done! Model trained and embeddings computed.")
