# Test Fixtures

The files in this directory are local fixtures used to keep the test suite
deterministic and offline-friendly.

## Downloaded mmCIF fixtures

Most `*.cif` files here are direct RCSB downloads saved with lowercase names, for
example:

```bash
curl -fL https://files.rcsb.org/download/7BZ5.cif -o tests/resources/7bz5.cif
curl -fL https://files.rcsb.org/download/5JG9.cif -o tests/resources/5jg9.cif
```

## Generated Chroma sample

`steps200_seed42_len100.cif` is a deterministic sample used by
`tests/models/test_chroma.py`. Regenerate it from the `chroma` Conda environment
with the public backbone and design weights available locally or via API access:

```bash
conda run -n chroma python -c "import torch; from chroma.models.chroma import Chroma; torch.manual_seed(42); chroma = Chroma('https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_backbone_v1.0.pt', 'https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_design_v1.0.pt', device='cpu'); protein = chroma.sample(chain_lengths=[100], steps=200); protein.to_CIF('tests/resources/steps200_seed42_len100.cif')"
```

If this fixture changes, update the expected ELBO constant in
`tests/models/test_chroma.py` to match the regenerated sample.
