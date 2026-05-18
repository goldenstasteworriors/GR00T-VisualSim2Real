# Door Asset Generation Scripts

Generate randomized articulated door USD assets offline, matching the same distributions used during training in `spawn_door()`.

<p align="center">
  <img src="../../../media/door_assets.gif" width="90%">
</p>

## Generate Door Assets

```bash
python gr00t/rl/scripts/generate_door_assets.py \
    --num_doors 100 \
    --output_dir data/door_assets \
    --build_latch \
    --add_floors \
    --door_open_lr right \
    --door_open_io out \
    --randomize_material \
    --seed 42
```

Each door is saved as a self-contained `.usd` file with randomized geometry (width, height, weight), handle placement, joint dynamics, optional latch, and Omniverse materials. A `metadata.json` with all sampled parameters is saved alongside the USD files.

Multiple values can be passed to `--door_open_lr` and `--door_open_io` to sample from both:

```bash
--door_open_lr left right --door_open_io in out
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--num_doors N` | Number of doors to generate | 100 |
| `--output_dir DIR` | Output directory | `data/door_assets` |
| `--seed S` | Random seed | 0 |
| `--build_latch` | Add latch mechanism | off |
| `--add_floors` | Add floor planes | off |
| `--add_walls` | Add surrounding walls | off |
| `--door_open_lr` | Hinge side(s): `left`, `right`, or both | `right` |
| `--door_open_io` | Open direction(s): `in`, `out`, or both | `out` |
| `--door_handle_tblr T B L R` | Handle position range (fraction of door) | `0.95 0.85 0.08 0.15` |
| `--randomize_material` | Apply random Omniverse materials | off |
| `--preloaded_materials_num_transform` | UV-transform variants per texture | 20 |
| `--preloaded_materials_num_color` | Random-color paint variants | 100 |

## Generate 1000 Doors (Batch)

A convenience script generates 1000 doors across 7 diverse configurations:

```bash
bash gr00t/rl/scripts/generate_1000_doors.sh            # default: data/door_assets/
bash gr00t/rl/scripts/generate_1000_doors.sh /my/path    # custom output dir
```

| Subdirectory | Count | Hinge | Direction | Latch | Walls |
|---|---|---|---|---|---|
| `right_out_latch/` | 200 | right | out | yes | no |
| `left_out_latch/` | 200 | left | out | yes | no |
| `right_in_latch/` | 150 | right | in | yes | no |
| `left_in_latch/` | 150 | left | in | yes | no |
| `mixed_walls/` | 100 | both | both | yes | yes |
| `mixed_no_latch/` | 100 | both | both | no | no |
| `wide_handle_range/` | 100 | both | both | yes | no |
