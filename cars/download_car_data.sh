# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

array=(001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020)

for num in "${array[@]}"; do
	var3=car_"$num".png
	wget "https://raw.githubusercontent.com/loliverhennigh/Steady-State-Flow-With-Neural-Nets/master/cars/$var3"
done