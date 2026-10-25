# Synthetic metadata fixtures

These 96×64 solid-color images were generated for RealOpen Lens regression tests.
The camera, lens, capture time, exposure settings and Paris GPS coordinates are
explicit test values; they do not describe a real photograph or device owner.

`camera-gps.jpg` was generated using Pillow's Image.Exif and IFDRational support.
`camera-gps.heic` was converted from it using the macOS command:

```sh
sips -s format heic camera-gps.jpg --out camera-gps.heic
```

The committed HEIC is a decoder fixture, so tests on Windows/Linux need no encoder.
The decoder and exact EXIF assertions run on both files. No external image license
is needed; these project-generated fixtures use the repository's Apache-2.0 terms.

`screenshot-marker.jpg` is a project-generated 320×192 terminal-like image with
EXIF UserComment `ASCII\0\0\0Screenshot` and a synthetic capture timestamp.
`screenshot-marker.heic` was converted using the same `sips` command above. The JPEG includes EXIF ImageDescription as well as UserComment. Conversion retains
the description as XMP.dc.description while dropping UserComment. Tests assert
both files have marker evidence and retain its actual source. The separately
committed `screenshot-stripped.heic` was converted from a UserComment-only JPEG:
it retains the timestamp but loses the marker and must remain unknown. These fixtures do not establish what any
particular iPhone/iOS AirDrop transfer retains; that depends on the received file.
