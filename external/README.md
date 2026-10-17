# Local inference source checkout

The official AlphaFold 3 source can be downloaded here as `alphafold3/`:

```powershell
git clone --depth 1 https://github.com/google-deepmind/alphafold3.git external/alphafold3
```

The checkout and any model parameters are excluded from this project's Git history. AlphaFold 3 remains a separate upstream project with its own source license and model-parameter terms.

The downloaded checkout was verified at upstream commit `03f4f48867748c164ae6d5144ae0529477df02e3`. Model weights have not been downloaded or used for inference.

Official setup instructions: <https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md>.

The official model download is <https://storage.googleapis.com/alphafold3/af3.bin.zst> (the server reported 1,020,545,840 compressed bytes when checked). A successful download is not evidence of successful inference. The documented GPU installation requires Linux; this Windows machine has an RTX 3050 with 4 GB VRAM. The upstream documentation also describes a slower, unsupported CPU inference mode. This project's short, MSA-free/template-free inputs avoid the full genetic database search, but still require a working AF3 runtime and model parameters.
