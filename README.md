# ecal

## Dependencies
- ceres
- hdf5
- opencv

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Run
```bash
./build/calibration_main config/calibration.yaml
```

## Docker
### One command (build + run)
```bash
bash run_docker.sh /absolute/path/to/your/data your_file.h5
```

The script does all of the following:
- builds image `ecal:latest`
- mounts `/absolute/path/to/your/data` as `/data`
- maps `/absolute/path/to/your/data/your_file.h5` to `/data/input.h5`
- disables GUI in container by default (`ECAL_ENABLE_GUI=0`)
- runs `calibration_main` with `/app/config/calibration.docker.yaml`

`config/calibration.docker.yaml` expects:
- input file: `/data/input.h5`
- output directory: `/data/results/`

Example host layout:
- `/absolute/path/to/your/data/your_file.h5`
- output will be written to `/absolute/path/to/your/data/results/`

Docker usage note:
- In Docker, real-time GUI windows are disabled by default.
- Visualization outputs are saved under `/data/results/` (host: `/absolute/path/to/your/data/results/`), so review them after the run finishes.

If file name is omitted, `input.h5` is used by default:
```bash
bash run_docker.sh /absolute/path/to/your/data
```

Enable GUI explicitly (only when display forwarding is configured):
```bash
ECAL_ENABLE_GUI=1 bash run_docker.sh /absolute/path/to/your/data your_file.h5
```
