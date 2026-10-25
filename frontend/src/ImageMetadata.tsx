import type { ExifMetadata } from "./api";

export function ImageMetadata({
  metadata,
}: {
  metadata?: Record<string, unknown>;
}) {
  const exif = metadata?.exif as ExifMetadata | undefined;
  if (!exif) return null;
  const exposure = exif.exposure_seconds;
  const fields = [
    [
      "Screenshot marker",
      exif.screenshot
        ? exif.screenshot.state === "marked"
          ? "Present"
          : "Not recorded"
        : null,
    ],
    [
      "Camera",
      [exif.camera_make, exif.camera_model].filter(Boolean).join(" · "),
    ],
    ["Lens", exif.lens_model],
    [
      "Taken",
      exif.captured_at
        ? `${exif.captured_at}${exif.capture_timezone_known ? "" : " (timezone not recorded)"}`
        : null,
    ],
    ["ISO", exif.iso],
    [
      "Exposure",
      exposure
        ? exposure < 1
          ? `1/${Math.round(1 / exposure)} s`
          : `${exposure} s`
        : null,
    ],
    ["Aperture", exif.aperture ? `f/${exif.aperture}` : null],
    [
      "Focal length",
      exif.focal_length_mm ? `${exif.focal_length_mm} mm` : null,
    ],
    [
      "GPS",
      exif.gps
        ? `${exif.gps.latitude.toFixed(6)}, ${exif.gps.longitude.toFixed(6)}`
        : null,
    ],
    [
      "Altitude",
      exif.gps?.altitude_m != null ? `${exif.gps.altitude_m} m` : null,
    ],
  ].filter(([, value]) => value != null && value !== "");
  return (
    <section aria-label="Photo metadata">
      <h3>Capture metadata</h3>
      {fields.length ? (
        <dl className="exif-fields">
          {fields.map(([key, value]) => (
            <div key={String(key)}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="muted">
          No camera, capture date or GPS recorded in this file.
        </p>
      )}
      {exif.screenshot?.state === "marked" && (
        <p className="muted">
          Evidence:{" "}
          {exif.screenshot.evidence.map((item) => item.source).join(" · ")}
        </p>
      )}
      {exif.warnings?.map((warning, index) => (
        <p className="issue" key={index}>
          {warning}
        </p>
      ))}
    </section>
  );
}
