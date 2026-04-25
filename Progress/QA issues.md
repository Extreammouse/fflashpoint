# QA Issues

## Open

- Final manual dense calibration validation is still needed on the cleaned codebase.
- Final manual runtime validation is still needed for:
  - `prototype_app.py --webcam`
  - `prototype_app.py --webcam --screen-flash --screen-flash-mode overlay`
  - `prototype_app.py --webcam --screen-flash --screen-flash-mode white`
- Final manual confirmation is still needed that no-face runtime fallback stays at zero confidence and never triggers flash.

## Resolved

- Cleaned the shipped surface down to `L2CS + torso detector + upper_torso trigger`.
- Removed abandoned camera-calibration and probe utilities from the maintained project surface.
- Removed MediaPipe and other fallback behavior from the maintained calibration/runtime path.
- Reduced `download_models.py` to the shipped model surface.
- Kept dense calibration and corner-aware model selection as the supported path.
- Rewrote the docs to match the actual runtime and calibration flow.
