python src/train.py \
    --config cfgs/models/convlstm_v1/convlstm_guillermo_v1_config.yaml \
    --trainer cfgs/trainers/trainer_test_short.yaml \
    --data cfgs/data/data_monotemporal_full_features.yaml \
    --data.batch_size 15 \
    --data.data_dir /path/to/data \
    --data.data_fold_id 0 \
    --do_test true