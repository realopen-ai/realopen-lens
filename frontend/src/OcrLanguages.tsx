import { useEffect, useId, useState } from "react";

export const DEFAULT_OCR_LANGUAGES = "eng+fra+ara";
export const DEFAULT_OCR_OPTIONS = {
  eng: "English",
  fra: "French",
  ara: "Arabic",
};

export function OcrLanguages({
  value,
  onChange,
  options = DEFAULT_OCR_OPTIONS,
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  options?: Record<string, string>;
  disabled?: boolean;
}) {
  const id = useId();
  const [manual, setManual] = useState(
    value && value !== "auto" ? value : DEFAULT_OCR_LANGUAGES,
  );
  useEffect(() => {
    if (value !== "auto") setManual(value);
  }, [value]);
  const automatic = value === "auto";
  const selected = (automatic ? manual : value).split("+").filter(Boolean);
  const choices = {
    ...options,
    ...Object.fromEntries(
      selected.filter((code) => !options[code]).map((code) => [code, code]),
    ),
  };
  return (
    <fieldset className="ocr-languages" disabled={disabled}>
      <legend>OCR languages</legend>
      <label className="ocr-mode">
        <input
          type="radio"
          name={id}
          checked={!automatic}
          onChange={() => onChange(manual || DEFAULT_OCR_LANGUAGES)}
        />
        Choose languages
      </label>
      <label className="ocr-mode">
        <input
          type="radio"
          name={id}
          checked={automatic}
          onChange={() => onChange("auto")}
        />
        Automatic script groups
      </label>
      {automatic ? (
        <p className="small muted">
          Detect writing scripts locally, then use their language packs.
          Uncertain detection uses all installed supported packs. Mixed scripts
          and short text can be missed.
        </p>
      ) : (
        <>
          <details className="ocr-language-choices">
            <summary>
              {selected.map((code) => options[code] || code).join(", ") ||
                "Select at least one language"}
            </summary>
            <div className="ocr-language-grid">
              {Object.entries(choices).map(([code, name]) => (
                <label className="checkbox" key={code}>
                  <input
                    type="checkbox"
                    checked={selected.includes(code)}
                    onChange={(event) =>
                      onChange(
                        (event.target.checked
                          ? [...selected, code]
                          : selected.filter((item) => item !== code)
                        ).join("+"),
                      )
                    }
                  />
                  {name}
                </label>
              ))}
            </div>
          </details>
          <label className="ocr-code-label">
            Language codes
            <input
              aria-label="Language codes"
              required
              pattern="[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*"
              value={value}
              onChange={(event) => onChange(event.target.value)}
            />
          </label>
          <p className="small muted">
            Enable the languages in your documents. Source installs need those
            packs installed.
          </p>
        </>
      )}
    </fieldset>
  );
}
