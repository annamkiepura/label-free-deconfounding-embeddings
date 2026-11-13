<!-- Generated from tableA2_encoders.tex by `python paper-artifacts/build.py tableA2_encoders`; the LaTeX file is the reference. -->

| Encoder | Hugging Face model | Dim. |
|---|---|---|
| MiniLM | sentence-transformers/all-MiniLM-L6-v2 | 384 |
| E5-small^a | intfloat/multilingual-e5-small | 384 |
| GIST-small | avsolatorio/GIST-small-Embedding-v0 | 384 |
| GIST-base | avsolatorio/GIST-Embedding-v0 | 768 |
| GIST-large | avsolatorio/GIST-large-Embedding-v0 | 1024 |
| E5-base^a | intfloat/multilingual-e5-base | 768 |
| E5-large^a | intfloat/multilingual-e5-large | 1024 |
| MPNet | sentence-transformers/all-mpnet-base-v2 | 768 |
| Nomic-v2^b | nomic-ai/nomic-embed-text-v2-moe | 768 |
| MXB | mixedbread-ai/mxbai-embed-large-v1 | 1024 |

Encoders (Sentence-Transformers, `trust_remote_code`; cosine similarity). Above the rule: also used in the primary experiments and OOD transfer; below: ten-encoder study only. In-distribution documents (and so queries) carry ^athe prefix `query:\textvisiblespace` or ^bthe named prompt `passage`, otherwise no prompt; OOD targets never do. Maximum sequence length: model default, not recorded. Recorded model revisions are in the accompanying data file.
