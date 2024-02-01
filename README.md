# Pracos: Project-specific Retrieval Augmentation for Better Code Summarization

## dependencies
Please refer to [knn-transformers](https://github.com/neulab/knn-transformers).
Additionally, you should install `pylucene` for subset retrieval.

## subset retrieval
cd to `subset_retrieval`, set `LANG`, `TEST_PROJECT`, `KEY`, and directory in `syntax_subset_retrieval.sh` and `semantics_subset_retrieval.sh`, then run them to perform sentence-level retrieval in history-based corpus construction.
The output is a file containing the ids in CSN train set. Merge the four files to get the union of all-type retrieved ids. And then construct the ext-subset from CSN and ids.

## datastore creation
First combine the retrieved ext-subset with project train set.
Then set the arguments in `datastore_creation.sh` and run `bash datastore_creation.sh` to create datastore.

## generation
### locality index file
First separately perform datastore creation for ext-subset and project train set. You can get `dstore_size.txt` in `dstore_dir`.
Then run `python get_locality_index.py` to get the locality index file.

### generate summaries
Set the arguments in `generation.sh` and run `bash generation.sh` to generate summaries.

## acknowledgement
The implementation is based on [knn-transformers](https://github.com/neulab/knn-transformers).