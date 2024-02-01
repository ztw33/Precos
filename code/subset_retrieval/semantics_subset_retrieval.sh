LANG=java
TEST_PROJECT=${LANG}/jodd
KEY=code  # code or nl

if [ $KEY == "code" ]; then
    MODEL=Salesforce/codet5-base-codexglue-sum-${LANG}
elif [ $KEY == "nl" ]; then
    MODEL=t5-base
else
    echo "Invalid KEY"
    exit 1
fi

# save datastore for CSN (only execute once for each language)
python semantics_subset_retrieval.py \
    --model_name_or_path ${MODEL} \
    --data_filename CSN-dataset/${LANG}/train.jsonl \
    --output_dir ../output_dir/CSN_dataset/${LANG}/subset-retrieval_sementics_${KEY} \
    --key ${KEY} \
    --save_dstore

# build index for CSN (only execute once for each language)
python semantics_subset_retrieval.py \
    --model_name_or_path ${MODEL} \
    --data_filename CSN-dataset/${LANG}/train.jsonl \
    --output_dir ../output_dir/CSN_dataset/${LANG}/subset-retrieval_sementics_${KEY} \
    --key ${KEY} \
    --build_index

# query for project dataset
python semantics_subset_retrieval.py \
    --model_name_or_path ${MODEL} \
    --data_filename ../../project_dataset/${TEST_PROJECT}/train.json \
    --dstore_dir ../output_dir/CSN_dataset/${LANG}/subset-retrieval_sementics_${KEY} \
    --output_dir ../output_dir/project_dataset/${TEST_PROJECT}/ext-subset_data/semantics_${KEY} \
    --subset \
    --key ${KEY} \
    --faiss_gpu \
    --topk=3
