# Table A2 (tableA2_encoders): configurations, sources and run IDs

Build: `python -m paper_assets build tableA2_encoders`. The LaTeX is rendered from `data/tableA2_encoders.json`, which this build derives from the saved runs. Run ID = `<path>@<sha256 prefix>`.

- Inventory: `inventory/encoder_inventory.json` (inventory/encoder_inventory.json@7f9709829af9)
- Code: `code/style_erasure_pipeline.py` (MODEL_MAP, `SentenceTransformer(id, trust_remote_code=True)`, no revision pin); `code/run_sae_experiment.py` `encode_texts` (document prompts); `code/ood.py` (`PROMPT_STYLE = "parent"`, target prompts, `encoder_revision`).

## MiniLM — ID documents

- model: sentence-transformers/all-MiniLM-L6-v2
- revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- revision_source: recorded by every OOD run (protocol.encoder_model); for the ID and encoder-study runs, the inventory's cache snapshot (same hash)
- dimension: 384
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the cached snapshot used by the runs
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra'], 'id': ['cap', 'scotus_oyez_lexis', 'scotus_oyez_lexis_wiki', 'semeval', 'swiltra'], 'ood_targets': ['5 MTEB legal retrieval tasks', 'STSBenchmark']}
- ood_donor_maps: ['cap', 'scotus_oyez_lexis', 'scotus_oyez_lexis_wiki']
- merged_with_ood_targets: OOD target prompts are empty, identical to the ID setting
- source files (26):
  - alternatives/generous_k_cap_mini.json@3856b131c350
  - alternatives/generous_k_scotus_oyez_lexis_mini.json@27ef7c457dd5
  - alternatives/generous_k_semeval_mini.json@cd6a1c66ce90
  - alternatives/generous_k_swiltra_mini.json@699b94dae21b
  - alternatives/geometry_cap_mini.json@be0c73ad7c3e
  - alternatives/geometry_scotus_oyez_lexis_mini.json@d6bea2e684fd
  - alternatives/geometry_scotus_oyez_lexis_wiki_mini.json@afa7455a1996
  - alternatives/geometry_semeval_mini.json@c485b85867bd
  - alternatives/geometry_swiltra_mini.json@ab0d8f13b88b
  - encoders/encoder_cap_mini.json@29df5cd05c92
  - encoders/encoder_scotus_oyez_lexis_mini.json@19f87a12b508
  - encoders/encoder_semeval_mini.json@9537543a4a47
  - encoders/encoder_swiltra_mini.json@dffe826f2b02
  - id/id_cap_mini.json@86843bb644d2
  - id/id_kmeans_scotus_oyez_lexis_mini.json@6e05d893adbf
  - id/id_kmeans_scotus_oyez_lexis_wiki_mini.json@48886d90557d
  - id/id_scotus_oyez_lexis_mini.json@22f8574e72e2
  - id/id_scotus_oyez_lexis_wiki_mini.json@8e072e36da63
  - alternatives/ood_kmeans_mini_cap.json@efc1c9306a8d
  - alternatives/ood_kmeans_mini_scotus_oyez_lexis.json@5e33660d5934
  - alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json@8c976488d560
  - alternatives/ood_ranks_mini_cap.json@578fb9ac4086
  - alternatives/ood_ranks_mini_scotus_oyez_lexis.json@c6991878dbae
  - ood/ood_mini_cap.json@edb3d5f96618
  - ood/ood_mini_scotus_oyez_lexis.json@9600062c4d06
  - ood/ood_mini_scotus_oyez_lexis_wiki.json@2fa727cbd1f7

## GIST-small — ID documents

- model: avsolatorio/GIST-small-Embedding-v0
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision 75e62fd210b9, excluded)
- dimension: 384
- pooling: cls_token
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_gist-small.json@78f98339a1e6
  - alternatives/generous_k_scotus_oyez_lexis_gist-small.json@ed850689340e
  - alternatives/generous_k_semeval_gist-small.json@35bb84de0761
  - alternatives/generous_k_swiltra_gist-small.json@ae4f1d617fed
  - alternatives/geometry_cap_gist-small.json@354d7856810e
  - alternatives/geometry_scotus_oyez_lexis_gist-small.json@15a872929577
  - alternatives/geometry_semeval_gist-small.json@7506626e1f65
  - alternatives/geometry_swiltra_gist-small.json@03cc83073ad1
  - encoders/encoder_cap_gist-small.json@b1102ce33d28
  - encoders/encoder_scotus_oyez_lexis_gist-small.json@3744dafe06e5
  - encoders/encoder_semeval_gist-small.json@f4030f87d580
  - encoders/encoder_swiltra_gist-small.json@1875c53d4ce4

## GIST-base — ID documents

- model: avsolatorio/GIST-Embedding-v0
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision bf6b2e55e92f, excluded)
- dimension: 768
- pooling: cls_token
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_gist-base.json@83f06fcdd9e4
  - alternatives/generous_k_scotus_oyez_lexis_gist-base.json@e84199da419d
  - alternatives/generous_k_semeval_gist-base.json@938c778bb8d9
  - alternatives/generous_k_swiltra_gist-base.json@e6005d454b1f
  - alternatives/geometry_cap_gist-base.json@6c11b62972bc
  - alternatives/geometry_scotus_oyez_lexis_gist-base.json@a8ba4bfce624
  - alternatives/geometry_semeval_gist-base.json@628d700a8269
  - alternatives/geometry_swiltra_gist-base.json@b4e0e68b10fe
  - encoders/encoder_cap_gist-base.json@530c3559d131
  - encoders/encoder_scotus_oyez_lexis_gist-base.json@ca2dcdedda55
  - encoders/encoder_semeval_gist-base.json@585dc4c76419
  - encoders/encoder_swiltra_gist-base.json@aad482e1df13

## GIST-large — ID documents

- model: avsolatorio/GIST-large-Embedding-v0
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision 7831200e2f78, excluded)
- dimension: 1024
- pooling: cls_token
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_gist-large.json@4bfbd7f16444
  - alternatives/generous_k_scotus_oyez_lexis_gist-large.json@e37b71d23c34
  - alternatives/generous_k_semeval_gist-large.json@edbea696b5ad
  - alternatives/generous_k_swiltra_gist-large.json@948398cbda35
  - alternatives/geometry_cap_gist-large.json@04febae22c2c
  - alternatives/geometry_scotus_oyez_lexis_gist-large.json@29bcb365efaf
  - alternatives/geometry_semeval_gist-large.json@666082a50540
  - alternatives/geometry_swiltra_gist-large.json@ecbd7b162c0a
  - encoders/encoder_cap_gist-large.json@8c226d5be69d
  - encoders/encoder_scotus_oyez_lexis_gist-large.json@8ec794c5cb46
  - encoders/encoder_semeval_gist-large.json@704aff4e2e0b
  - encoders/encoder_swiltra_gist-large.json@c5f991fc2457

## E5-small — ID documents

- model: intfloat/multilingual-e5-small
- revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
- revision_source: inventory: model-cache snapshot attributed to these runs; not saved in the ID or encoder-study result files
- dimension: 384
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the cached snapshot used by the runs
- document_prompt: "query: " on every document
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'id': ['bills_tweets', 'cap', 'scotus_oyez_lexis', 'scotus_oyez_lexis_wiki', 'scotus_oyez_wiki', 'scotus_wiki_lexis', 'semeval', 'swiltra', 'swiltra_de_fr_it', 'swiltra_de_it', 'swiltra_fr_it'], 'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: ['cap', 'scotus_oyez_lexis', 'scotus_oyez_lexis_wiki', 'swiltra', 'swiltra_de_fr_it']
- source files (51):
  - alternatives/alternatives_cap_e5-small.json@25e1ebccd17c
  - alternatives/alternatives_scotus_oyez_lexis_e5-small.json@f878c04bea85
  - alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json@cd001e4cd530
  - alternatives/alternatives_semeval_e5-small.json@a9a5102c5ffc
  - alternatives/alternatives_swiltra_de_fr_it_e5-small.json@d7ccdea02eb3
  - alternatives/alternatives_swiltra_e5-small.json@209bdcec521e
  - alternatives/generous_k_cap_e5-small.json@8bec3dd189b3
  - alternatives/generous_k_scotus_oyez_lexis_e5-small.json@6fbf637abeef
  - alternatives/generous_k_semeval_e5-small.json@e4c47bb482b7
  - alternatives/generous_k_swiltra_e5-small.json@94d5205d0cef
  - alternatives/geometry_cap_e5-small.json@cec5c41fde9a
  - alternatives/geometry_scotus_oyez_lexis_e5-small.json@1db57cd30499
  - alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json@906bec2d6e3b
  - alternatives/geometry_semeval_e5-small.json@c421a8318d74
  - alternatives/geometry_swiltra_de_fr_it_e5-small.json@9bf1607d9278
  - alternatives/geometry_swiltra_e5-small.json@35e90866a3dd
  - alternatives/kmeans_count_scotus_oyez_lexis_e5-small.json@e466c6ac09a1
  - alternatives/kmeans_count_scotus_oyez_lexis_wiki_e5-small.json@b233a987270e
  - alternatives/kmeans_count_swiltra_de_fr_it_e5-small.json@a9b0dec81d8d
  - alternatives/kmeans_count_swiltra_e5-small.json@cd3ec1236603
  - encoders/encoder_cap_e5-small.json@144f54d5b9da
  - encoders/encoder_scotus_oyez_lexis_e5-small.json@103f8bae69b1
  - encoders/encoder_semeval_e5-small.json@e15b8f24ec4c
  - encoders/encoder_swiltra_e5-small.json@eb3646e65b4b
  - id/id_bills_tweets_e5-small.json@defab91a806f
  - id/id_cap_e5-small.json@fda1e808c3f0
  - id/id_kmeans_bills_tweets_e5-small.json@408f188c1d86
  - id/id_kmeans_cap_e5-small.json@7bfa4fa2a36e
  - id/id_kmeans_scotus_oyez_lexis_e5-small.json@075f0996e4a2
  - id/id_kmeans_scotus_oyez_lexis_wiki_e5-small.json@b0a24f1184ba
  - id/id_kmeans_scotus_oyez_wiki_e5-small.json@ceeeb6c83eaf
  - id/id_kmeans_scotus_wiki_lexis_e5-small.json@fed3bce2cf73
  - id/id_kmeans_semeval_e5-small.json@82195d2b2508
  - id/id_kmeans_swiltra_de_fr_it_e5-small.json@c6cbc49b5dfd
  - id/id_kmeans_swiltra_de_it_e5-small.json@0eec7df6e313
  - id/id_kmeans_swiltra_e5-small.json@7dc74fb3dfcf
  - id/id_kmeans_swiltra_fr_it_e5-small.json@cb5f1fdc7846
  - id/id_scotus_oyez_lexis_e5-small.json@019aa95905db
  - id/id_scotus_oyez_lexis_wiki_e5-small.json@640167d23e06
  - id/id_scotus_oyez_wiki_e5-small.json@2649dfa86950
  - id/id_scotus_wiki_lexis_e5-small.json@e6a1512951e2
  - id/id_semeval_e5-small.json@5b7591702466
  - id/id_swiltra_de_fr_it_e5-small.json@9dfe0c6ffb1f
  - id/id_swiltra_de_it_e5-small.json@fddb308902ba
  - id/id_swiltra_e5-small.json@539061857e6d
  - id/id_swiltra_fr_it_e5-small.json@c373cfda1f5a
  - mechanism/mechanism_scotus_oyez_lexis_wiki_e5-small.json@a1c2c5748fb9
  - mechanism/mechanism_semeval_e5-small.json@cbeef28aba40
  - mechanism/mechanism_swiltra_de_fr_it_e5-small.json@008e3ac45e62
  - random_directions/random_directions_scotus_oyez_lexis_wiki_e5-small.json@e8c80c10ac43
  - random_directions/random_directions_swiltra_de_fr_it_e5-small.json@15bfb249f913

## E5-small — OOD targets

- model: intfloat/multilingual-e5-small
- revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
- revision_source: recorded by every OOD run (protocol.encoder_model)
- dimension: 384
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the cached snapshot used by the runs
- document_prompt: none
- query_prompt: none
- sts_prompt: none
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'ood_targets': ['5 MTEB legal retrieval tasks', 'STSBenchmark']}
- ood_donors: ['cap', 'scotus_oyez_lexis', 'scotus_oyez_lexis_wiki', 'swiltra', 'swiltra_de_fr_it']
- source files (13):
  - alternatives/ood_kmeans_e5-small_cap.json@71c5bdf22ce0
  - alternatives/ood_kmeans_e5-small_scotus_oyez_lexis.json@5dd0d1959f2b
  - alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json@3e5608d82b0d
  - alternatives/ood_kmeans_e5-small_swiltra.json@3ecb21d39113
  - alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json@a7c80cd184a9
  - alternatives/ood_ranks_e5-small_cap.json@e097e32f2650
  - alternatives/ood_ranks_e5-small_scotus_oyez_lexis.json@121c53d4a8f5
  - alternatives/ood_ranks_e5-small_swiltra.json@0a15cef15fc1
  - ood/ood_e5-small_cap.json@ed5186373151
  - ood/ood_e5-small_scotus_oyez_lexis.json@9d4af1052978
  - ood/ood_e5-small_scotus_oyez_lexis_wiki.json@a828cc98034b
  - ood/ood_e5-small_swiltra.json@63d3d726ecfb
  - ood/ood_e5-small_swiltra_de_fr_it.json@816e622be73c

## E5-base — ID documents

- model: intfloat/multilingual-e5-base
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision d12875059715, excluded)
- dimension: 768
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: "query: " on every document
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_e5-base.json@dc2ffc2b97d3
  - alternatives/generous_k_scotus_oyez_lexis_e5-base.json@686a0ad63303
  - alternatives/generous_k_semeval_e5-base.json@dbaeedfbccaa
  - alternatives/generous_k_swiltra_e5-base.json@e248f30e0f28
  - alternatives/geometry_cap_e5-base.json@6c79cb627f22
  - alternatives/geometry_scotus_oyez_lexis_e5-base.json@bf0c791ae688
  - alternatives/geometry_semeval_e5-base.json@c21ef84f7a5d
  - alternatives/geometry_swiltra_e5-base.json@e7e5a671e531
  - encoders/encoder_cap_e5-base.json@be19cf6b8a67
  - encoders/encoder_scotus_oyez_lexis_e5-base.json@b6fbf2eca864
  - encoders/encoder_semeval_e5-base.json@c4c964ef023e
  - encoders/encoder_swiltra_e5-base.json@2ecc3aeedb00

## E5-large — ID documents

- model: intfloat/multilingual-e5-large
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision 3d7cfbdacd47, excluded)
- dimension: 1024
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: "query: " on every document
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_e5-large.json@3ea287eef9c3
  - alternatives/generous_k_scotus_oyez_lexis_e5-large.json@57ea67a6564d
  - alternatives/generous_k_semeval_e5-large.json@5cc481914a76
  - alternatives/generous_k_swiltra_e5-large.json@04f756358982
  - alternatives/geometry_cap_e5-large.json@b666babc66f6
  - alternatives/geometry_scotus_oyez_lexis_e5-large.json@cea7b79e7462
  - alternatives/geometry_semeval_e5-large.json@a4bc0ccc2da8
  - alternatives/geometry_swiltra_e5-large.json@0872faf3b897
  - encoders/encoder_cap_e5-large.json@fd529ac00ce5
  - encoders/encoder_scotus_oyez_lexis_e5-large.json@43ef343a4d8e
  - encoders/encoder_semeval_e5-large.json@693891116ade
  - encoders/encoder_swiltra_e5-large.json@7d3079b93030

## MPNet — ID documents

- model: sentence-transformers/all-mpnet-base-v2
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision e8c3b32edf54, excluded)
- dimension: 768
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_mpnet.json@2d69834eb4a5
  - alternatives/generous_k_scotus_oyez_lexis_mpnet.json@4cd58d4c0ee9
  - alternatives/generous_k_semeval_mpnet.json@c22d3963c1f0
  - alternatives/generous_k_swiltra_mpnet.json@19010ff1c808
  - alternatives/geometry_cap_mpnet.json@7022c6e4dd20
  - alternatives/geometry_scotus_oyez_lexis_mpnet.json@104abb5aae50
  - alternatives/geometry_semeval_mpnet.json@f7602ccdf93d
  - alternatives/geometry_swiltra_mpnet.json@e3ec83f02d02
  - encoders/encoder_cap_mpnet.json@e3ab8d435ba0
  - encoders/encoder_scotus_oyez_lexis_mpnet.json@7b1ebeb16c73
  - encoders/encoder_semeval_mpnet.json@0a41388772fb
  - encoders/encoder_swiltra_mpnet.json@79f3a128295b

## Nomic-v2 — ID documents

- model: nomic-ai/nomic-embed-text-v2-moe
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision 1066b6599d09, excluded)
- dimension: 768
- pooling: mean
- normalization: L2 (Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: prompt_name="passage"
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_nomic-v2.json@3e039b492f1d
  - alternatives/generous_k_scotus_oyez_lexis_nomic-v2.json@abea2eac4bc5
  - alternatives/generous_k_semeval_nomic-v2.json@8ede00a250d3
  - alternatives/generous_k_swiltra_nomic-v2.json@780ded2c4aeb
  - alternatives/geometry_cap_nomic-v2.json@f9dd9cd38ffa
  - alternatives/geometry_scotus_oyez_lexis_nomic-v2.json@cfddaa7c555a
  - alternatives/geometry_semeval_nomic-v2.json@a13b242e5603
  - alternatives/geometry_swiltra_nomic-v2.json@e51a8f75de20
  - encoders/encoder_cap_nomic-v2.json@d4ee971b2443
  - encoders/encoder_scotus_oyez_lexis_nomic-v2.json@3d972bfd3b23
  - encoders/encoder_semeval_nomic-v2.json@3f56182c573f
  - encoders/encoder_swiltra_nomic-v2.json@13d58bcaebe4

## MXB — ID documents

- model: mixedbread-ai/mxbai-embed-large-v1
- revision: not recorded
- revision_source: not recorded (inventory holds only the current Hub revision b33106f585b9, excluded)
- dimension: 1024
- pooling: cls_token
- normalization: none (no Normalize module)
- pooling_normalization_source: inventory: modules of the current Hub configuration (not run-attributed)
- document_prompt: none
- query_prompt: same as document (symmetric: every query is a document)
- max_length: not recorded
- max_length_note: max_seq_length is not set in code (model default); the value was not saved by any run
- coverage: {'study': ['cap', 'scotus_oyez_lexis', 'semeval', 'swiltra']}
- ood_donor_maps: []
- source files (12):
  - alternatives/generous_k_cap_mxb.json@479f72ed708b
  - alternatives/generous_k_scotus_oyez_lexis_mxb.json@c6eedd13c542
  - alternatives/generous_k_semeval_mxb.json@e94dc928b988
  - alternatives/generous_k_swiltra_mxb.json@2e8cc18f0ad4
  - alternatives/geometry_cap_mxb.json@e0831f686017
  - alternatives/geometry_scotus_oyez_lexis_mxb.json@888e1b9d2d90
  - alternatives/geometry_semeval_mxb.json@2bd9d07a24ec
  - alternatives/geometry_swiltra_mxb.json@f90acee2c3f0
  - encoders/encoder_cap_mxb.json@d0fb182908f0
  - encoders/encoder_scotus_oyez_lexis_mxb.json@feb57403d305
  - encoders/encoder_semeval_mxb.json@e46ca7356ff8
  - encoders/encoder_swiltra_mxb.json@86175a5a49c9

## Not recorded (listed, not filled)

- MiniLM (ID documents): max_length
- GIST-small (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- GIST-base (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- GIST-large (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- E5-small (ID documents): max_length
- E5-small (OOD targets): max_length
- E5-base (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- E5-large (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- MPNet (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- Nomic-v2 (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- MXB (ID documents): revision, max_length, run-attributed pooling/normalization (current configuration shown)
- No run records the Sentence-Transformers or transformers version used for the ID and encoder-study embeddings; OOD runs record the mteb version (2.20.11).

## Checks (32 passed)

- the saved runs use exactly the ten registry encoders (['e5-base', 'e5-large', 'e5-small', 'gist-base', 'gist-large', 'gist-small', 'mini', 'mpnet', 'mxb', 'nomic-v2'])
- mini: inventory model id sentence-transformers/all-MiniLM-L6-v2 = code MODEL_MAP
- mini: every result file that saves `dim` has 384 (4 files)
- mini: all 8 OOD files record one model revision and one target-prompt setting
- mini: OOD runs record model sentence-transformers/all-MiniLM-L6-v2
- mini: OOD-recorded revision 1110a243fdf4 = the inventory's cache snapshot
- gist-small: inventory model id avsolatorio/GIST-small-Embedding-v0 = code MODEL_MAP
- gist-small: every result file that saves `dim` has 384 (4 files)
- gist-base: inventory model id avsolatorio/GIST-Embedding-v0 = code MODEL_MAP
- gist-base: every result file that saves `dim` has 768 (4 files)
- gist-large: inventory model id avsolatorio/GIST-large-Embedding-v0 = code MODEL_MAP
- gist-large: every result file that saves `dim` has 1024 (4 files)
- e5-small: inventory model id intfloat/multilingual-e5-small = code MODEL_MAP
- e5-small: every result file that saves `dim` has 384 (4 files)
- e5-small: all 13 OOD files record one model revision and one target-prompt setting
- e5-small: OOD runs record model intfloat/multilingual-e5-small
- e5-small: OOD-recorded revision 614241f622f5 = the inventory's cache snapshot
- e5-base: inventory model id intfloat/multilingual-e5-base = code MODEL_MAP
- e5-base: every result file that saves `dim` has 768 (4 files)
- e5-large: inventory model id intfloat/multilingual-e5-large = code MODEL_MAP
- e5-large: every result file that saves `dim` has 1024 (4 files)
- mpnet: inventory model id sentence-transformers/all-mpnet-base-v2 = code MODEL_MAP
- mpnet: every result file that saves `dim` has 768 (4 files)
- nomic-v2: inventory model id nomic-ai/nomic-embed-text-v2-moe = code MODEL_MAP
- nomic-v2: every result file that saves `dim` has 768 (4 files)
- mxb: inventory model id mixedbread-ai/mxbai-embed-large-v1 = code MODEL_MAP
- mxb: every result file that saves `dim` has 1024 (4 files)
- OOD targets are encoded without any prompt for every OOD encoder (['E5-small', 'MiniLM']), as the caption states
- every in-distribution document prompt has a caption mark (['"query: " on every document', 'none', 'prompt_name="passage"'])
- tableA2_encoders: every citation and reference resolves
- tableA2_encoders: the float fits the TACL text block
- tableA2_encoders: no overfull boxes in the TACL preview

## Issues

- **warning** `a2_revision_not_recorded`: gist-small: model revision not recorded by any run (inventory fallback 75e62fd210b9 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: gist-base: model revision not recorded by any run (inventory fallback bf6b2e55e92f is the current Hub revision)
- **warning** `a2_revision_not_recorded`: gist-large: model revision not recorded by any run (inventory fallback 7831200e2f78 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: e5-base: model revision not recorded by any run (inventory fallback d12875059715 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: e5-large: model revision not recorded by any run (inventory fallback 3d7cfbdacd47 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: mpnet: model revision not recorded by any run (inventory fallback e8c3b32edf54 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: nomic-v2: model revision not recorded by any run (inventory fallback 1066b6599d09 is the current Hub revision)
- **warning** `a2_revision_not_recorded`: mxb: model revision not recorded by any run (inventory fallback b33106f585b9 is the current Hub revision)
- **warning** `tacl_style_standin`: writing/tacl_analysis_draft/tacl2021v1.sty not found: measured on a stand-in page with the TACL geometry (455.24pt x 702.8pt text, 219.09pt columns, Times)
