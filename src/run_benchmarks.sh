# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# ** PLEASE NOTE: **
# In each of the commands, the script output is being dumped into the file "$OUTPUT_FILE". 
# This is to avoid cmd line output which is temporary and available only as long as the window is open. 
# Having it in an output file allows the user to refer back to it.
# As a result, the cmd window will be frozen. Please be patient as the scripts are executed sequentially and track updates in $OUTPUT_FILE

#!/usr/bin/env bash
set -e

OUTPUT_FILE="$OUTPUT_DIR/output.txt"
echo "output_file" > $OUTPUT_FILE

python ./src/train_TF_model.py -b 4 -e 5 -m $OUTPUT_DIR/models/batch_size_4 -l $OUTPUT_DIR/logs/batch_size_4.log -lossf ./losses/batch_size_4_intel.csv >> $OUTPUT_FILE
python ./src/train_TF_model.py -b 8 -e 5 -m $OUTPUT_DIR/models/batch_size_8 -l $OUTPUT_DIR/logs/batch_size_8.log -lossf ./losses/batch_size_8_intel.csv >> $OUTPUT_FILE
python ./src/train_TF_model.py -b 16 -e 5 -m $OUTPUT_DIR/models/batch_size_16 -l $OUTPUT_DIR/logs/batch_size_16.log -lossf ./losses/batch_size_16_intel.csv >> $OUTPUT_FILE
python ./src/train_TF_model.py -b 32 -e 5 -m $OUTPUT_DIR/models/batch_size_32 -l $OUTPUT_DIR/logs/batch_size_32.log -lossf ./losses/batch_size_32_intel.csv >> $OUTPUT_FILE

python ./src/convert_keras_to_frozen_graph.py -s $OUTPUT_DIR/models/batch_size_4 -o $OUTPUT_DIR/models/batch_size_4_frozen >> $OUTPUT_FILE

python ./src/test_TF_model.py -b 1 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -l $OUTPUT_DIR/logs/batch_size_1_inf.log -lossf ./losses/batch_size_1_loss.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 4 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -l $OUTPUT_DIR/logs/batch_size_4_inf.log -lossf ./losses/batch_size_4_loss.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 8 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -l $OUTPUT_DIR/logs/batch_size_8_inf.log -lossf ./losses/batch_size_8_loss.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 16 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -l $OUTPUT_DIR/logs/batch_size_16_inf.log -lossf ./losses/batch_size_16_loss.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 32 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -l $OUTPUT_DIR/logs/batch_size_32_inf.log -lossf ./losses/batch_size_32_loss.csv >> $OUTPUT_FILE

python ./src/TF_model_INC_quantization.py -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model.pb -o $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/quantization.log

python ./src/test_TF_model.py -b 1 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/batch_size_1_inf_INC.log -lossf ./losses/batch_size_1_loss_INC.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 4 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/batch_size_4_inf_INC.log -lossf ./losses/batch_size_4_loss_INC.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 8 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/batch_size_8_inf_INC.log -lossf ./losses/batch_size_8_loss_INC.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 16 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/batch_size_16_inf_INC.log -lossf ./losses/batch_size_16_loss_INC.csv >> $OUTPUT_FILE
python ./src/test_TF_model.py -b 32 -m $OUTPUT_DIR/models/batch_size_4_frozen/saved_frozen_model_INC.pb -l $OUTPUT_DIR/logs/batch_size_32_inf_INC.log -lossf ./losses/batch_size_32_loss_INC.csv >> $OUTPUT_FILE
