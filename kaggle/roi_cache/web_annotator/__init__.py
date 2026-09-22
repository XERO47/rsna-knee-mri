"""Remote ROI landmark annotation application.

See ``kaggle/roi_cache/web_annotator/README.md`` and
``references/REMOTE_ROI_ANNOTATION_APP_SPEC.md`` for the contract this
package implements. This package never reimplements DICOM geometry, series
selection, pixel decoding, or patient-coordinate conversion; it imports those
primitives from ``roi_geometry`` and ``build_roi_cache``.
"""
