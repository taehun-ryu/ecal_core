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
