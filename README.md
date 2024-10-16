# MS2Data4DOLPHYN

MS2Data4DOLPHYN represnets <u>M</u>utli-<u>S</u>ource <u>M</u>ulti-<u>S</u>cale <u>Data</u> for <u>DOLPHYN</u>, which is aimed to integrate multi-source and multi scale data into some energy system optimization models like [DOLPHYN.jl](https://github.com/macroenergy/Dolphyn.jl) using large language models like GPT and large climate models like [Pangu](https://www.nature.com/articles/s41586-023-06185-3).

## ERA5 data

## Global weather forecasting (inference) using the Pangu models

#### Downloading trained models

Please download the four pre-trained models (~1.1GB each) from Google drive or Baidu netdisk:

The 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1M7SAigVsCSH8hpw6DE8TDQ?pwd=ie0h)

The 3-hour model (pangu_weather_3.onnx): [Google drive](https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/197fZsoiCqZYzKwM7tyRrfg?pwd=gmcl)

The 6-hour model (pangu_weather_6.onnx): [Google drive](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3)

The 24-hour model (pangu_weather_24.onnx): [Google drive](https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy)

These models are stored using the ONNX format, and thus can be used via different languages such as Python, C++, C#, Java, etc.

#### Input data preparation using Python

Please prepare the input data using [numpy](https://numpy.org/). There are two files that shall be put under the `input_data` folder, namely, `input_surface.npy` and `input_upper.npy`.

`input_surface.npy` stores the input surface variables. It is a numpy array shaped (4,721,1440) where the first dimension represents the 4 surface variables (MSLP, U10, V10, T2M **in the exact order**).

`input_upper.npy` stores the upper-air variables. It is a numpy array shaped (5,13,721,1440) where the first dimension represents the 5 surface variables (Z, Q, T, U and V **in the exact order**), and the second dimension represents the 13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa **in the exact order**).

In both cases, the dimensions of 721 and 1440 represent the size along the latitude and longitude, where the numerical range is [90,-90] degree and [0,359.75] degree, respectively, and the spacing is 0.25 degrees. For each 721x1440 slice, the data format is exactly the same as the `.nc` file download from the ERA5 official website.

Note that the numpy arrays should be in single precision (`.astype(np.float32)`), not in double precision.

We support ERA5 initial fields and ECMWF initial fields (e.g., the initial fields of the HRES forecast), where the latter often leads to a slight accuracy drop (mainly for T2M because the two fields are quite different in temperature). A `.nc` file of ERA5 can be transformed into a `.npy` file using the netCDF4 package, and a `.grib` file of the ECMWF initial fields can be transformed into a `.npy` file using the pygrib package. Note that Z represents geopotential, not geopotential height, so a factor of 9.80665 should be multiplied if the raw data contains the geopotential height.

We temporarily do not support other kinds of initial fields due to the possibly dramatic differences in the fields when Z<0.

We provide an example of transferred input files, `input_surface.npy` and `input_upper.npy`, which correspond to the ERA5 initial fields of at 12:00UTC, 2018/09/27. Please download them from Google drive or Baidu netdisk:

`input_surface.npy`: [Google drive](https://drive.google.com/file/d/1pj8QEVNpC1FyJfUabDpV4oU3NpSe0BkD/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1i4o5i8guAqmOus6PWncAlA?pwd=4z9s)

`input_upper.npy`: [Google drive](https://drive.google.com/file/d/1--7xEBJt79E3oixizr8oFmK_haDE77SS/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1mS8X5MqEdbVfF2u2Us62FQ?pwd=sgx6)

#### Inference

After the above steps are finished, please check `inference_cpu.py` for an example of making a 24-hour weather forecast on CPU with the 24-hour model, and `inference_gpu.py` for the GPU version.

For example, running the following command, one can get the 24-hour forecast in the `output_data` folder:
```
python inference_cpu.py # python inference_gpu.py for gpu environment
```

Also, `inference_iterative.py` shows an example to generate per-6-hour forecast within a week.

## Pseudocode and how to use

`pseudocode.py` contains the pseudocode that elaborates our main algorithm. It is written in Python and can be implemented using any deep learning library, e.g. PyTorch and TensorFlow.

Note that one needs to download about 60TB of ERA5 data and prepare for computational resource of 3000 GPU-days (in V100) to train each model.

## Renewable potential estimation from weather data

## Heat demand estimation from weather data

## DOLPHYN integration

## LLM-based DOLPHYN interaction