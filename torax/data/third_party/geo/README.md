```
Copyright 2024 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This folder contains third-party geometry files for various tokamaks.


## Scenarios

### ITER hybrid scenarios

This scenario corresponds to an ITER hybrid scenario simulated in CRONOS,
as presented in J. Citrin et al 2010 Nucl. Fusion 50 115007.
The equilibria were generated using [CHEASE](https://gitlab.epfl.ch/spc/chease).
The canonical source of truth for this data is [here](https://gitlab.com/qualikiz-group/pyntegrated_model/-/tree/4a3656ffbcb92ab166f255d9f62d8722cb15d5fd/geo).

The mat2cols file is [`pyntegrated_model/geo/ITER_hybrid_citrin_equil_cheasedata.mat2cols`](https://gitlab.com/qualikiz-group/pyntegrated_model/-/blob/4a3656ffbcb92ab166f255d9f62d8722cb15d5fd/geo/ITER_hybrid_citrin_equil_cheasedata.mat2cols).

The EQDSK files were generated from [`pyntegrated_model/geo/EQDSK_ITERhybrid_COCOS02`](https://gitlab.com/qualikiz-group/pyntegrated_model/-/blob/4a3656ffbcb92ab166f255d9f62d8722cb15d5fd/geo/EQDSK_ITERhybrid_COCOS02) as follows:

```
# Download
wget https://gitlab.com/qualikiz-group/pyntegrated_model/-/raw/main/geo/EQDSK_ITERhybrid_COCOS02

# Convert to COCOS11 using CHEASE functions
matlab -nodisplay
>> addpath(genpath('chease'));
>> eqdsk_cocos02 = read_eqdsk('EQDSK_ITERhybrid_COCOS02', 2);
>> [eqdsk_cocos11] = eqdsk_cocos_transform(eqdsk_cocos02, [2, 11]);
>> write_eqdsk('EQDSK_ITERhybrid_COCOS11_chease.eqdsk', eqdsk_cocos11, [11]);
>> exit

# The output file we want is EQDSK_ITERhybrid_COCOS11_chease.eqdsk_COCOS11_IpB0positive
# Trim off comments, which are unsupported by eqdsk python package
python3
>>> with open("EQDSK_ITERhybrid_COCOS11_chease.eqdsk_COCOS11_IpB0positive", "r") as f1:
...    trimmed_eqdsk = f1.read().split(' \n')[0]
...    with open("iterhybrid_cocos11.eqdsk", "w") as f2:
...       f2.write(trimmed_eqdsk)

# We also trim the COCOS02 version for comparison / testing
>>> with open("EQDSK_ITERhybrid_COCOS02", "r") as f1:
...    trimmed_eqdsk = f1.read().split(' \n')[0]
...    with open("iterhybrid_cocos02.eqdsk", "w") as f2:
...       f2.write(trimmed_eqdsk)
```


## Licensing

All associated LICENSE files are also located in this directory.
The mapping of datafile: LICENSE for the geometry files in this directory are as follows.

| File                     | License      |
| ------------------------ | ------------ |
| iterhybrid.mat2cols      | LICENSE_PINT |
| iterhybrid_cocos02.eqdsk | LICENSE_PINT |
| iterhybrid_cocos11.eqdsk | LICENSE_PINT |
