# Web annotator: browser UI, crosshair, and coordinate mapping

Source: `REMOTE_ROI_ANNOTATION_APP_SPEC.md` (section 5) and
`WEB_ANNOTATOR_REFERENCE.md` (static UI contract, function inventory, and the
2026-09-21 display-quality correction).

## HTML/CSS/JS constraints

Use one static HTML page, one CSS file, and one small JavaScript file. No
frontend build system is needed. The visual design can resemble the current
Matplotlib window, but a Matplotlib GUI cannot itself be delivered to a remote
browser. Use an HTML canvas or image element with an accurately positioned
overlay marker.

As implemented: `kaggle/roi_cache/web_annotator/static/{index.html,app.css,app.js}`.
One page, no build step, no framework. `app.js` owns exactly one piece of
scientific-adjacent logic — the CSS-pixel → source-pixel coordinate transform
for clicks — and nothing else; it never computes a patient coordinate. It
treats every server response as authoritative and re-fetches `/api/progress`
after every save to decide whether to show the completion screen, rather than
tracking completion state itself.

## 5.1 Header

Display:

- cohort name: `Pilot` or `Final`;
- reader display name;
- `Patient X/Y` where Y is that reader's assigned patient count;
- `N patients remaining`;
- `Landmark 1/4`, `2/4`, `3/4`, or `4/4`;
- a human-readable landmark name;
- four task-state indicators for the current patient: incomplete, saved, or
  unable-to-annotate;
- save state: `Unsaved point`, `Saving…`, `Saved`, or `Offline—not saved`.

Do not show `1/80` or `1/400` as the primary progress measure. Do not call a
study a task. Full UIDs may be available in a collapsed technical-details panel
for troubleshooting, but not as the main title.

## 5.2 Image area

- Show one grayscale slice centered on a dark neutral background.
- Preserve the source aspect ratio.
- Do not mirror, rotate, or horizontally flip the image.
- Show the selected point as a high-contrast crosshair that does not obscure
  the anatomy.
- Show slice position as `Slice X/Y`.
- Provide a range slider and previous/next slice buttons.
- Keyboard left/right changes the slice. Changing slice clears any unsaved
  point and asks for confirmation if necessary.
- Provide simple zoom-in, zoom-out, reset, and pan controls. Zooming must not
  alter saved source pixel coordinates.
- Do not show an old marker after changing slices.

## 5.3 Click mapping (exact coordinate transform)

The client must map a pointer location through the displayed image transform
back to fractional source pixel coordinates. Mapping must account for CSS
scaling, aspect-ratio letterboxing, zoom, pan, and high-DPI screens. A click
outside the actual image is rejected.

The save request includes the task identity, selected SOP UID or opaque slice
identifier, and fractional row/column. The server confirms:

1. the task is assigned to the authenticated reader;
2. the study, landmark, series, and slice are the server's current resolved
   entities, not arbitrary client values;
3. row and column are finite and inside source bounds;
4. pixel-to-patient-to-pixel error is at most `1e-5` pixels;
5. the task revision has not changed since it was loaded.

Only then may the record be committed.

## 5.4 Controls

Required controls:

- Previous patient
- Previous landmark
- Save and next landmark
- Next landmark
- Next patient
- Replace saved point
- Unable to annotate
- Reset view

`Save and next landmark` must be the primary action. The button is disabled
until a valid point exists. Replacing a saved point requires an explicit second
save and creates an audit event; it must not silently overwrite history.

`Unable to annotate` requires one of a short fixed list plus optional notes
(`config.UNABLE_REASONS`, also read by `schemas.UnableRequest` and
`static/index.html`'s `<select>` so both share one source):

- anatomy not confidently visible;
- reference series unsuitable;
- corrupt/unreadable image;
- laterality concern;
- other.

These records remain incomplete for scientific export until reviewed. They
must not be exported as `qc_status=pass` and must not silently become zero
coordinates.

## 5.5 Instructions

Each landmark panel should show a short, agreed radiologist instruction and an
optional reviewed example image. The UI must not improvise clinical guidance.
The examples and wording require sign-off from both radiologists before the
pilot begins.

## 5.6 Completion screen

When all assigned work is saved or explicitly flagged, show:

- saved patient count;
- unresolved/flagged task count;
- a list of patients needing review;
- confirmation that data is already stored on the Mac;
- a safe sign-out button.

Do not rely on a final `Submit all` operation. Every individual save is already
durable.

## Current visual contract (as implemented)

- The four per-patient chips always show anatomical names, never raw states
  such as `incomplete`: medial joint-line center, lateral joint-line center,
  patellofemoral center, and suprapatellar recess.
- A completed (`pass`) chip turns green. An incomplete chip remains neutral,
  an unable chip remains yellow, and the active chip has a blue outline.
- `drawCrosshair` renders a target reticle in CSS-pixel units so Retina scaling
  cannot make it tiny: 34 px arms, a 10 px radius circle, a center dot, a
  5 px black contrast under-stroke, and a 2.5 px colored stroke. Unsaved points
  are red and stored points are green.
- Static CSS/JS URLs carry a cache-busting `?v=N` query so a normal page
  refresh fetches the current UI revision instead of reusing an older browser
  cache. Current version is `v=4` (see the display-quality correction below).
- Scientific click coordinates are unchanged across all visual revisions: the
  reticle is display-only and its center remains exactly the fractional source
  pixel sent to the server.

## Important static UI functions

| Symbol | Input | Output/behavior | Purpose |
|---|---|---|---|
| `landmarkLabel` | machine landmark name | reviewed display title or fallback | Keeps machine identifiers out of the radiologist-facing chips. |
| `renderHeader` | authoritative task response | patient/landmark header and four chips | Uses `landmarkLabel(entry.landmark)`; state controls CSS color only and is never printed as the chip label. |
| `drawScene` | image plus pending/saved point | canvas frame | Red target for pending selection, green target for persisted point. |
| `drawCrosshair` | backing-store x/y and color | high-contrast target reticle | Multiplies all dimensions by device pixel ratio so the physical CSS size is stable. |
| `sourceToCanvas` / `canvasToSource` | source or canvas coordinates | inverse mapped coordinates | Accounts for letterboxing, zoom, pan, and backing-store scaling; marker styling does not participate in coordinate math. |
| `clientToCanvasCoordinates` | pointer event | backing-store x/y | Corrects CSS scaling and high-DPI canvas dimensions. |

## 2026-09-21: real final-cohort save incident and UI correction

A normal click on patient 5 of the final cohort returned HTTP `400` while the
server was still live. The browser UI incorrectly labelled every non-2xx save
response as **Offline**. The true cause was a server-side geometry inverse bug
(see `../roi_cache/laterality_rules.md` for the `patient_to_pixel` fix); the
UI-side correction is documented here because it changes user-facing behavior:

`static/app.js` now treats a real network exception as **Offline** only. A
server-side rejected click remains selected, becomes **Point not saved**, and
shows a yellow explanation; it must be adjusted or marked unable rather than
silently retried as though connectivity failed. `index.html` uses cache
version `v=3` for this change, so refresh after a server restart loads this
behavior.

## 2026-09-21: web display-quality correction

The web client returns the same lossless 8-bit PNG generated from the desktop
annotator's own `_display_pixels` transform; no DICOM pixel decoding, modality
LUT, percentile windowing, source resolution, or click-coordinate math changed.
The visual issue was entirely canvas-side: `drawScene` had
`ctx.imageSmoothingEnabled = false`, which nearest-neighbour enlarged common
384×384 source MR images to the 900-pixel viewport and made them appear
pixelated. It now sets `imageSmoothingEnabled = true` and
`imageSmoothingQuality = "high"`. The update is cache-busted as static asset
version `v=4`; a normal browser refresh loads it without a server restart.
`test_enlarged_mri_slices_use_high_quality_canvas_scaling` protects this
radiologist-facing rendering contract.

## Left-knee joint-line correction UI

The correction app exposes the DICOM laterality and display orientation at the
top of every task. For a coronal joint-line task it states, for example,
`LEFT KNEE. In this displayed coronal image, MEDIAL is on the LEFT side.` The
instruction is calculated from that exact series' DICOM patient-coordinate
mapping; it is withheld rather than guessed for a non-coronal or too-oblique
image. This guidance is specifically for `medial_joint_line` and
`lateral_joint_line`; PF and suprapatellar tasks are centreline landmarks and
carry no side instruction.

Reopening a task for correction never destroys the old point: it is retained
in the append-only `annotation_events` history and the app renders it amber
while the candidate correction is red. On save, the new point becomes the live
green point and the previous point remains auditable.
