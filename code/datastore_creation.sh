LANG=java
TEST_PROJECT=${LANG}/geopackage-android
TRAIN_FILE=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/train.json
OUTPUT_DIR=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types
DSTORE_DIR=${OUTPUT_DIR}/dstore_dir

TOKENIZER=Salesforce/codet5-base
MODEL=Salesforce/codet5-base-codexglue-sum-${LANG}

# save datastore
python -u run_translation.py  \
  --model_name_or_path ${MODEL} --tokenizer_name ${TOKENIZER} \
  --train_file ${TRAIN_FILE} --validation_file ${TRAIN_FILE} \
  --per_device_train_batch_size=32 --per_device_eval_batch_size=32 \
  --output_dir ${OUTPUT_DIR} \
  --source_lang code --target_lang nl \
  --dstore_dir ${DSTORE_DIR} \
  --save_knnlm_dstore --do_eval --eval_subset train 

# build index
python -u run_translation.py  \
  --model_name_or_path ${MODEL} --tokenizer_name ${TOKENIZER} \
  --train_file ${TRAIN_FILE} --validation_file ${TRAIN_FILE} \
  --per_device_train_batch_size=32 --per_device_eval_batch_size=32 \
  --output_dir ${OUTPUT_DIR} \
  --source_lang code --target_lang nl \
  --dstore_dir ${DSTORE_DIR} \
  --build_index 


