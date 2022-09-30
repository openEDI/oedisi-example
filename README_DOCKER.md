# Running NREL State Estimator Docker

First, download `gadal-example.tar.gz`.

Then run `docker load -i gadal-example.tar.gz`

To get a docker volume pointed at the right place locally, we have to create a directory and point a volume.
```
mkdir outputs_build
docker volume create --name gadal_output --opt type=none --opt device=$(PWD)/outputs_build --opt o=bind
```

If `pwd` is unavailable on your system, then you must specify the exact path. On windows, this will end up
being `/c/Users/.../outputs_builds/`. You must use forward slashes.

Then we can run the docker image:
```
docker run --rm --mount source=gadal_output,target=/simulation/outputs gadal-example:0.0.0
```

You can omit the docker volume parts as well as `--mount` if you do not care about the exact outputs.

## Analysis

The `outputs_build` directory should contain `.feather` and `.csv` files describing the output.

- `voltage_real` and `voltage_imag` are the "true" voltages from OpenDSS
- `voltage_mag` and `voltage_angle` are the outputs from the state estimator

The `post_analysis.py` script computes relative error for magnitude (MAPE) and angle in percentage (MAE). Plots of errors over time and bus number is also produced.

To run on the correct docker file, use
```
python post_analysis.py outputs_build
```
