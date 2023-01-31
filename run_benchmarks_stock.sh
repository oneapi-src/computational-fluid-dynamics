# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# ** PLEASE NOTE: **
# In each of the commands, the script output is being dumped into the file "output.txt". 
# This is to avoid cmd line output which is temporary and available only as long as the window is open. 
# Having it in an output allows the user to refer back to it.
# As a result, the cmd window will be frozen. Please be patient as the scripts are executed sequentially and track updates in output.txt

#!/bin/bash

echo "output_file" > output.txt

python ./src/train_TF_model.py -b 4 -e 5 -m ./models/stock/batch_size_4 -l ./logs/stock/batch_size_4.log -lossf ./losses/stock/batch_size_4_intel.csv >> output.txt
python ./src/train_TF_model.py -b 8 -e 5 -m ./models/stock/batch_size_8 -l ./logs/stock/batch_size_8.log -lossf ./losses/stock/batch_size_8_intel.csv >> output.txt
python ./src/train_TF_model.py -b 16 -e 5 -m ./models/stock/batch_size_16 -l ./logs/stock/batch_size_16.log -lossf ./losses/stock/batch_size_16_intel.csv >> output.txt
python ./src/train_TF_model.py -b 32 -e 5 -m ./models/stock/batch_size_32 -l ./logs/stock/batch_size_32.log -lossf ./losses/stock/batch_size_32_intel.csv >> output.txt

python ./src/convert_keras_to_frozen_graph.py -s ./models/stock/batch_size_4 -o ./models/stock/batch_size_4_frozen >> output.txt

python ./src/test_TF_model.py -b 1 -m ./models/stock/batch_size_4_frozen/saved_frozen_model.pb -l ./logs/stock/batch_size_1_inf.log -lossf ./losses/stock/batch_size_1_loss.csv >> output.txt
python ./src/test_TF_model.py -b 4 -m ./models/stock/batch_size_4_frozen/saved_frozen_model.pb -l ./logs/stock/batch_size_4_inf.log -lossf ./losses/stock/batch_size_4_loss.csv >> output.txt
python ./src/test_TF_model.py -b 8 -m ./models/stock/batch_size_4_frozen/saved_frozen_model.pb -l ./logs/stock/batch_size_8_inf.log -lossf ./losses/stock/batch_size_8_loss.csv >> output.txt
python ./src/test_TF_model.py -b 16 -m ./models/stock/batch_size_4_frozen/saved_frozen_model.pb -l ./logs/stock/batch_size_16_inf.log -lossf ./losses/stock/batch_size_16_loss.csv >> output.txt
python ./src/test_TF_model.py -b 32 -m ./models/stock/batch_size_4_frozen/saved_frozen_model.pb -l ./logs/stock/batch_size_32_inf.log -lossf ./losses/stock/batch_size_32_loss.csv >> output.txt