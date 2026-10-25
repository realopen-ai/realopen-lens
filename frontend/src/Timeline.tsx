import type { ImageResult } from "./api";

function calendar(value: string, options: Intl.DateTimeFormatOptions) {
  // Display the recorded calendar day without converting it into the browser's zone.
  return new Intl.DateTimeFormat(undefined, {
    ...options,
    timeZone: "UTC",
  }).format(new Date(`${value}T12:00:00Z`));
}

export function TimelineHeading({
  image,
  previous,
}: {
  image: ImageResult;
  previous?: ImageResult;
}) {
  const day = image.timeline?.day;
  const month = image.timeline?.month;
  const newMonth = month !== previous?.timeline?.month || !previous;
  const newDay = day !== previous?.timeline?.day || !previous;
  if (!newDay) return null;
  return (
    <div className="timeline-heading">
      {newMonth && (
        <h3>
          {month && day
            ? calendar(`${month}-01`, { month: "long", year: "numeric" })
            : "Date unknown"}
        </h3>
      )}
      {day && (
        <h4>
          {calendar(day, { weekday: "long", month: "short", day: "numeric" })}
        </h4>
      )}
    </div>
  );
}

export function TimelineDate({ image }: { image: ImageResult }) {
  const timeline = image.timeline;
  const label =
    timeline?.source === "capture"
      ? "Captured"
      : timeline?.source === "modified"
        ? "File modified"
        : "Date unknown";
  return (
    <p
      className="timeline-date"
      title={
        timeline?.source === "modified"
          ? "No capture date recorded. File modification date shown in UTC."
          : timeline?.source === "capture"
            ? "Capture date as recorded in image metadata."
            : "No usable date recorded."
      }
    >
      {label}
      {timeline?.day
        ? ` · ${calendar(timeline.day, { month: "short", day: "numeric", year: "numeric" })}`
        : ""}
    </p>
  );
}
