# Code contributions

We welcome contributions to the Chroma code base, including new conditioners, integrators, patches, bug fixes, and others.

Note that your contributions will be governed by the Apache 2.0 license, meaning that you will be giving us permission to use your contributed code under the conditions specified in the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) (also available in [LICENSE.txt](LICENSE.txt)).

## How to Contribute

Please use GitHub pull requests to contribute code. See
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests. We will try to monitor incoming requests with some regularity, but cannot promise a specific timeframe within which we will review your request. 

## Tests and Fixtures

Use the `chroma` Conda environment for local testing. The repository CI workflow in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the data and utility,
model, layer, integration, and release smoke suites separately. The shared
cross-platform CI environment is [`environment.yml`](environment.yml), while
[`environment_Mac.yaml`](environment_Mac.yaml) keeps the Apple Silicon local setup
used during development. CI installs from
[`environment.linux-64.lock`](environment.linux-64.lock), and Apple Silicon
local development should use
[`environment.osx-arm64.lock`](environment.osx-arm64.lock).

To regenerate the explicit lockfiles from [`environment.yml`](environment.yml),
run:

```bash
python scripts/generate_lockfiles.py
```

The Linux lock targets `__glibc=2.17`, which matches the minimum ABI required
by the pinned `linux-64` packages and is sufficient for the GitHub Actions
`ubuntu-latest` runners used by CI.

Fixture provenance and regeneration notes live in
[`tests/resources/README.md`](tests/resources/README.md), including the generated
`steps200_seed42_len100.cif` sample used by `tests/models/test_chroma.py`.

For quick local iteration, you can skip the heavier tiers with:

```bash
pytest -m "not integration and not slow"
```
