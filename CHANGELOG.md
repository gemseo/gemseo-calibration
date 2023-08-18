<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 2.0.0 (June 2023)

Update to GEMSEO 5.

### Changed

- Data are `dict[str, ndarray]` objects as
    `{variable_name: variable_values}` instead of
    `~gemseo.datasets.dataset.Dataset`{.interpreted-text role="class"}.
- Use `""` as empty value of `str` and `str | Path` arguments, instead
    of `"None"`.
- `.CalibrationMeasure`{.interpreted-text role="class"}: the type of
    `f_type` is
    `~gemseo.core.mdofunctions.mdo_function.MDOFunction.FunctionType`{.interpreted-text
    role="attr"}.

## Version 1.0.0 (July 2022)

First release.
