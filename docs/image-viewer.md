# Image viewer

Click an image or its filename in the local gallery to open the viewer. It works
with folder browsing, text/image search and virtual collections, including results
from multiple folders.

- **Previous/next** buttons and the left/right arrow keys navigate the currently
  loaded images in their displayed order. Load another gallery page before opening
  the viewer to include more images. Navigation does not widen the search scope.
- **Fit** fits the image to the canvas. Zoom controls switch to native 100%, 150%
  and 200% display; scroll inside the canvas to pan. Each image starts in Fit mode.
- The information panel shows dimensions, image format, source folder, paths and
  processing issues. **Extracted text** loads the full text retained in the indexed
  generation rather than the gallery's 300-character preview. It does not run OCR
  again or recover text omitted by indexing limits.
- **Open original** opens the first available, content-verified copy in the local
  system's default image application. **Reveal in folder** selects it in Finder
  or Explorer; on Linux it opens the containing directory using `xdg-open`.
- **Find similar images** uses the existing folder/library/virtual-collection scope.
- **Relevant/Irrelevant** updates the same saved search as the gallery controls.
  Feedback requires an explicitly recorded search; browsing alone does not create
  training labels. Exact-query memory retains its existing behavior, including
  hiding an irrelevant result and making Undo available in the gallery.
- Virtual-collection checkboxes show groups for the image's source folder and
  workspace-wide groups. They add or remove membership without modifying files.
  Cached exact-pixel aliases count as the same image; near-duplicate reviews do
  not imply shared membership. A conflicting collection revision is reported and
  requires an explicit retry.

Escape or **Close** dismisses the viewer and restores focus to its opening control
when that control still exists. Focus stays within the dialog while it is open.
The viewer adapts to light/dark themes and stacks the image above details on small
screens. If the browser cannot display the original, the indexed thumbnail is used
with a visible notice. Missing detail data and desktop-action failures remain visible.

## Local API

`GET /api/image-details` accepts `image_id`, `collection`, optional `collection_id`
and optional `generation`. A stable collection ID takes precedence over its name,
so library results remain associated with their actual source after renaming.
It returns indexed metadata, full retained OCR text, source availability and virtual
collection membership. It reads an immutable generation and performs no inference.

`POST /api/image-actions` accepts the same reference and `action: "open" | "reveal"`.
The original is resolved inside the indexed folder and its bytes must match the
image ID before launching the system handler. Client-provided filesystem paths or
commands are not accepted. Both endpoints use the gallery's existing local-session
and origin checks. Desktop actions require a graphical session on the server's
machine; headless failures are reported instead of claiming success.
