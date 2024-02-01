TOKENIZER=Salesforce/codet5-base
K=64

LANG=java
MODEL=Salesforce/codet5-base-codexglue-sum-${LANG}
PROJECT_LIST=("geopackage-android" "jain-slee" "jboss-common-core" "jodd" "lojix" "OpenEstate-IO" "orientdb" "thredds" "TieFaces" "wildfly" "aeron" "boon" "Fluid" "GeoRegression" "logback-android" "spring-security" "parquet-mr" "wro4j")
KNN_TEMP_LIST=(10 5 5 20 5 5 5 10 20 20 10 10 50 5 10 10 20 20)
LMBDA_LIST=(0.4 0.4 0.3 0.4 0.3 0.1 0.5 0.5 0.5 0.2 0.4 0.3 0.8 0.3 0.3 0.6 0.5 0.4)
W_LIST=(1.3 0.65 0.95 0.95 0.95 1.15 1.1 0.6 0.75 1.15 0.75 1.05 1.05 0.65 0.55 1.1 0.5 0.85)

for ((i=0; i<${#PROJECT_LIST[@]}; i++)); do
  TEST_PROJECT=${PROJECT_LIST[$i]}
  KNN_TEMP=${KNN_TEMP_LIST[$i]}
  LMBDA=${LMBDA_LIST[$i]}
  W=${W_LIST[$i]}

  TRAIN_FILE=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/train.json
  TEST_FILE=../project_dataset/${TEST_PROJECT}/test.json
  OUTPUT_DIR=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_all-types_proj
  DSTORE_DIR=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/dstore_dir
  LOCALITY_FILE=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/locality_index.json

  python -u run_translation.py  \
    --model_name_or_path ${MODEL} --tokenizer_name ${TOKENIZER} \
    --train_file ${TRAIN_FILE} --validation_file ${TEST_FILE} \
    --per_device_eval_batch_size=4 \
    --output_dir ${OUTPUT_DIR} \
    --source_lang code --target_lang nl \
    --do_eval \
    --predict_with_generate \
    --dstore_dir ${DSTORE_DIR} \
    --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
    --knn \
    --include_locality \
    --locality_file ${LOCALITY_FILE} \
    --locality_weight ${W}
done


LANG=python
MODEL=Salesforce/codet5-base-codexglue-sum-${LANG}
PROJECT_LIST=("airflow" "grimoirelab-perceval" "h2o-3" "probability" "pylint" "pypet" "qiskit-terra" "flowcraft" "hwt" "PyFunceble" "PyKMIP" "pyopenssl" "pyrser" "vaex" "DeepPavlov" "sregistry-cli")
KNN_TEMP_LIST=(5 20 50 20 20 5 20 10 20 10 50 20 10 20 20 5)
LMBDA_LIST=(0.3 0.5 0.4 0.5 0.2 0.4 0.5 0.6 0.6 0.4 0.7 0.6 0.2 0.6 0.5 0.6)
W_LIST=(1.45 1.0 0.55 0.85 0.7 0.75 0.75 0.55 1.1 0.7 1.15 1.35 1.25 0.8 1.5 0.75)

for ((i=0; i<${#PROJECT_LIST[@]}; i++)); do
  TEST_PROJECT=${PROJECT_LIST[$i]}
  KNN_TEMP=${KNN_TEMP_LIST[$i]}
  LMBDA=${LMBDA_LIST[$i]}
  W=${W_LIST[$i]}

  TRAIN_FILE=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/train.json
  TEST_FILE=../project_dataset/${TEST_PROJECT}/test.json
  OUTPUT_DIR=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_all-types_proj
  DSTORE_DIR=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/dstore_dir
  LOCALITY_FILE=output_dir/project_dataset/${TEST_PROJECT}/ext-subset_proj_data/all-types/locality_index.json

  python -u run_translation.py  \
    --model_name_or_path ${MODEL} --tokenizer_name ${TOKENIZER} \
    --train_file ${TRAIN_FILE} --validation_file ${TEST_FILE} \
    --per_device_eval_batch_size=4 \
    --output_dir ${OUTPUT_DIR} \
    --source_lang code --target_lang nl \
    --do_eval \
    --predict_with_generate \
    --dstore_dir ${DSTORE_DIR} \
    --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
    --knn \
    --include_locality \
    --locality_file ${LOCALITY_FILE} \
    --locality_weight ${W}
done

