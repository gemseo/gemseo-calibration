<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 4.0.0 (August 2025)

### Added

- The [signal][gemseo_calibration.signal] package
  proposes new features to facilitate the calibration of a signal generator,
  such as an [ODEDiscipline][gemseo.disciplines.ode].
- The package [problems.signal][gemseo_calibration.problems.signal]
  proposes problems to illustrate the features of the [signal][gemseo_calibration.signal] package.
- [Calibrator][gemseo_calibration.calibrator.Calibrator]
  has a `formulation_settings_model` argument and keyword arguments `**formulation_settings`
  (use either one or the other).
- [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
  has a `formulation_settings_model` argument and keyword arguments `**formulation_settings`
  (use either one or the other).

### Changed

- [DataVersusModel][gemseo_calibration.post.data_versus_model.post.DataVersusModel]
  displays data on a grid.
- [MultipleScatter][gemseo_calibration.post.multiple_scatter.MultipleScatter]
  displays data on a grid.
- API CHANGES:
    - `BaseCalibrationMetric.mesh` renamed to [CalibrationMetricSettings.mesh_name][gemseo_calibration.metrics.settings.CalibrationMetricSettings].
    - use the expression _calibration metric_ rather the _calibration measure_,
      to be consistent with [gemseo.utils.metrics][gemseo.utils.metrics].
    - rename `measures` package to [metrics][gemseo_calibration.metrics].
    - rename `gemseo_calibration.measure.CalibrationMeasure` to [gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric].
    - `gemseo_calibration.calibrator.CalibrationMeasure`:
        - rename it to [gemseo_calibration.metrics.settings.CalibrationMetricSettings][gemseo_calibration.metrics.settings.CalibrationMetricSettings].
        - [CalibrationMetricSettings][gemseo_calibration.metrics.settings.CalibrationMetricSettings] is a Pydantic model and so the arguments must be set as keyword arguments.
        - move it to [gemseo_calibration.metrics.settings][gemseo_calibration.metrics.settings].
        - rename its field `measure` to `metric`.
        - rename its field `mesh` to `mesh_name`.
        - rename its field `output` to `output_name`.
    - rename `IntegratedMeasure` to [BaseIntegratedMetric][gemseo_calibration.metrics.base_integrated_metric.BaseIntegratedMetric].
    - rename `MeanMeasure` to [BaseMeanMetric][gemseo_calibration.metrics.base_mean_metric.BaseMeanMetric].
    - [Calibrator][gemseo_calibration.calibrator.Calibrator]:
        - rename its `maximize_objective_measure` attribute to [maximize_objective_metric][gemseo_calibration.calibrator.Calibrator.maximize_objective_metric].
        - rename its `add_measure` method to [add_metric][gemseo_calibration.calibrator.Calibrator.add_metric].
        - rename its `formulation` argument to `formulation_name`
        - rename its `control_outputs` argument to `metric_settings_models`.
    - `CalibrationMeasureFactory`:
        - rename it to [CalibrationMetricFactory][gemseo_calibration.metrics.factory.CalibrationMetricFactory].
        - remove its `measures` property; use `class_names` instead.
        - rename its method `is_integrated_measure` to [is_integrated_metric][gemseo_calibration.metrics.factory.CalibrationMetricFactory.is_integrated_metric].
    - remove the `formulation` argument of [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]; use `formulation_name` instead.
    - rename the `control_outputs` argument of [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario] to `metric_settings_models`.

## Version 3.0.0 (November 2024)

### Added

- Support GEMSEO v6.
- Support for Python 3.12.

## Version 2.0.2 (December 2023)

### Added

- Support for Python 3.11.

### Removed

- Support for Python 3.8.

## Version 2.0.1 (September 2023)

### Fixed

- Compatibility with recent versions of NumPy.

## Version 2.0.0 (June 2023)

- Support GEMSEO v5.

### Changed

- Data are `dict[str, ndarray]` objects a `{variable_name: variable_values}` instead of
  [Dataset][gemseo.datasets.dataset.Dataset].
- Use `""` as empty value of `str` and `str | Path` arguments, instead of `"None"`.
- [BaseCalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric]:
  the type of `f_type` is [FunctionType][gemseo.core.mdo_functions.mdo_function.MDOFunction.FunctionType].

## Version 1.0.0 (July 2022)

First release.
