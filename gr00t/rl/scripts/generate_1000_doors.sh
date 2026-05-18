#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate 1000 door USD assets with diverse configurations.
#
# Usage:
#   bash gr00t/rl/scripts/generate_1000_doors.sh
#   bash gr00t/rl/scripts/generate_1000_doors.sh /path/to/custom/output_dir

set -e

BASE_DIR="${1:-data/door_assets}"
SCRIPT="gr00t/rl/scripts/generate_door_assets.py"

echo "Generating 1000 door assets in ${BASE_DIR}/ ..."

# Config 1 (200 doors): Training default — right hinge, opens out, with latch & floors
python $SCRIPT \
    --num_doors 200 \
    --output_dir "${BASE_DIR}/right_out_latch" \
    --build_latch --add_floors \
    --door_open_lr right --door_open_io out \
    --randomize_material --seed 0

# Config 2 (200 doors): Left hinge, opens out, with latch & floors
python $SCRIPT \
    --num_doors 200 \
    --output_dir "${BASE_DIR}/left_out_latch" \
    --build_latch --add_floors \
    --door_open_lr left --door_open_io out \
    --randomize_material --seed 100

# Config 3 (150 doors): Right hinge, opens in, with latch & floors
python $SCRIPT \
    --num_doors 150 \
    --output_dir "${BASE_DIR}/right_in_latch" \
    --build_latch --add_floors \
    --door_open_lr right --door_open_io in \
    --randomize_material --seed 200

# Config 4 (150 doors): Left hinge, opens in, with latch & floors
python $SCRIPT \
    --num_doors 150 \
    --output_dir "${BASE_DIR}/left_in_latch" \
    --build_latch --add_floors \
    --door_open_lr left --door_open_io in \
    --randomize_material --seed 300

# Config 5 (100 doors): Mixed directions, with walls & latch
python $SCRIPT \
    --num_doors 100 \
    --output_dir "${BASE_DIR}/mixed_walls" \
    --build_latch --add_floors --add_walls \
    --door_open_lr left right --door_open_io in out \
    --randomize_material --seed 400

# Config 6 (100 doors): Mixed directions, no latch
python $SCRIPT \
    --num_doors 100 \
    --output_dir "${BASE_DIR}/mixed_no_latch" \
    --add_floors \
    --door_open_lr left right --door_open_io in out \
    --randomize_material --seed 500

# Config 7 (100 doors): Wider handle height range for diversity
python $SCRIPT \
    --num_doors 100 \
    --output_dir "${BASE_DIR}/wide_handle_range" \
    --build_latch --add_floors \
    --door_open_lr left right --door_open_io in out \
    --door_handle_tblr 0.98 0.75 0.06 0.20 \
    --randomize_material --seed 600

echo ""
echo "Done! Generated 1000 door assets across 7 configurations in ${BASE_DIR}/"
echo ""
echo "  right_out_latch/    (200)  Training default: right hinge, opens outward"
echo "  left_out_latch/     (200)  Left hinge, opens outward"
echo "  right_in_latch/     (150)  Right hinge, opens inward"
echo "  left_in_latch/      (150)  Left hinge, opens inward"
echo "  mixed_walls/        (100)  Mixed directions with surrounding walls"
echo "  mixed_no_latch/     (100)  Mixed directions without latch mechanism"
echo "  wide_handle_range/  (100)  Wider handle placement range"
