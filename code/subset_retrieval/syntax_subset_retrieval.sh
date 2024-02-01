LANG=java
TEST_PROJECT=${LANG}/jodd
KEY=code  # code or nl

# build index for CSN (only execute once for each language)
python syntax_subset_retrieval.py \
    --build_index \
    --key ${KEY} \
    --data_filename CSN_dataset/${LANG}/train.json \
    --output_dir ../output_dir/CSN_dataset/${LANG}/subset-retrieval_syntax_${KEY}

# query
python syntax_subset_retrieval.py \
    --subset \
    --key ${KEY} \
    --data_filename ../../project_dataset/${TEST_PROJECT}/train.json \
    --output_dir ../output_dir/project_dataset/${TEST_PROJECT}/ext-subset_data/syntax_${KEY} \
    --index_dir ../output_dir/CSN_dataset/${LANG}/subset-retrieval_syntax_${KEY} \
    --topk 3
