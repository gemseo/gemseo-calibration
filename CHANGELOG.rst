..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Changelog titles are:
   - Added for new features.
   - Changed for changes in existing functionality.
   - Deprecated for soon-to-be removed features.
   - Removed for now removed features.
   - Fixed for any bug fixes.
   - Security in case of vulnerabilities.

Changelog
=========

All notable changes of this project will be documented here.

The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version 2.0.0 (May 2023)
************************

Changed
-------

- Data are ``dict[str, ndarray]`` objects as ``{variable_name: variable_values}`` instead of :class:`~gemseo.datasets.dataset.Dataset`.
- Use ``""`` as empty value of ``str`` and ``str | Path`` arguments, instead of ``"None"``.
- :class:`.CalibrationMeasure`: the type of ``f_type`` is :attr:`~gemseo.core.mdofunctions.mdo_function.MDOFunction.FunctionType`.

Version 1.0.0 (July 2022)
*************************

First release.
