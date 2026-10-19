# Optional local GLM-OCR

This is a retained experiment. Current pipeline development and text-region
evaluation use [Tesseract only](text-regions.md). The optional adapter remains
available for reproducing the earlier comparison.

Tesseract is the default OCR backend. GLM-OCR is explicitly selectable through an
installed local Ollama. RapidOCR remains a benchmark-only experiment.

## Run an image

Download the model explicitly once, using your installed Ollama:

```sh
ollama pull glm-ocr:q8_0
```

With Ollama running, use the installed project CLI:

```sh
lens ocr data/example.png
lens ocr data/example.png --engine glm-ocr
```

The first command uses Tesseract. The second uses the local Q8 model through
`http://127.0.0.1:11434/api/generate`. It does not require the GLM-OCR SDK,
Transformers, PyTorch, a cloud key or an additional Python client dependency.
Existing ONNX inference remains available for embeddings and the RapidOCR trial;
GLM-OCR uses Ollama's model runtime instead.

Options include `--model`, `--ollama-port`, `--timeout`, `--max-tokens` and
`--context`. Model download is a separate setup operation. OCR never pulls models
implicitly. A missing model produces an explicit error. Selecting GLM does not
silently fall back to Tesseract, so benchmark failures remain attributable.

The backend uses numeric loopback HTTP directly. Proxy environment variables and
redirects are not used. Installed model identity and vision support are checked;
cloud model metadata is rejected. The default generation profile is:

- Task prompt: `Text Recognition:`.
- Temperature 0, fixed seed 20261019, two CPU inference threads.
- Context 8192 tokens and output budget 4096 tokens.
- Input capped to 2400px on its longest side after the shared safe decoder.
- Ninety-second HTTP timeout; benchmark supervision adds a hard item deadline.
- One request at a time; only the requested model is unloaded at batch/CLI exit.

Ollama may use Metal on Apple Silicon; these settings are not a hard two-core CPU
limit. Weight size is not peak inference memory. The shared 64MiB encoded/24MP
decoded image limits still apply. Only the first page/frame is processed, with
an explicit warning for multipage inputs.

Output retains the generated text verbatim. The backend does not invent word
boxes or confidence values. `done_reason=length` is recorded as a truncation
warning. Token counts and server durations are retained, with units declared.
A generative model can reorder text, insert formatting or generate unsupported
content; successful execution is not proof of faithful transcription.

Exactly uniform RGB pixels are checked before resizing and skip inference, with
empty text and an explicit `inference_skipped=uniform_pixels` result. This narrow
guard does not detect general text-free scenes or prevent all false positives.

See the [completed three-way 500-image comparison](benchmarks/2026-10-19-three-way-ocr.md)
for measured quality, latency, memory, omissions and hallucination examples.

## Three-engine benchmark

Install optional RapidOCR tooling for the three-way comparison:

```sh
uv sync --locked --group ocr-eval --no-editable
```

Run all engines sequentially on the same dataset:

```sh
uv run --no-sync python -m benchmarks.ocr_compare --dataset data/benchmark-500-v2 --output runs/ocr-comparison/three-engines --engines tesseract rapidocr glm-ocr --threads 2
```

The runner checks every image checksum and uses the same audited references and
quality metrics for all engines. Tesseract and RapidOCR run on CPU; Ollama may use
Metal. Engine hardware differences are recorded rather than presented as a CPU
microbenchmark. Initialization/warmup is excluded from individual image latency
and retained separately. Generated text is scored as returned, without removing
markup just to increase scores.

The same command resumes the same frozen run. Configuration, model digest,
package versions, evaluation identity and source fingerprints must match. Old
runs remain intact; use a new output directory after configuration/code changes.
Every image result, warning/error/timeout and available accuracy metric is retained
in SQLite and per-engine JSONL. Pairwise changes are written for every engine pair.
Untranscribed images are not counted as accurate, and expected decode-policy
rejections are separated from inferred images.

Separate engine runs can be combined after validating all shared conditions:

```sh
uv run --no-sync python -m benchmarks.ocr_compare --engines tesseract rapidocr --output runs/ocr-comparison/controls
uv run --no-sync python -m benchmarks.ocr_compare --engines glm-ocr --output runs/ocr-comparison/glm
uv run --no-sync python -m benchmarks.ocr_compare --merge runs/ocr-comparison/controls runs/ocr-comparison/glm --output runs/ocr-comparison/combined
```

The merger rejects duplicate engines, missing cases and mismatched image manifests,
references, decoder/evaluation code, dependencies, host, language or thread settings.
Component configuration hashes and original run paths remain in the combined report.

## Memory and offline validation

Ollama inference runs outside the Python worker. The benchmark therefore samples
the Python process tree **plus the Ollama server and its model-runner descendants**.
With multiple local servers, specify `--ollama-server-pid` to attribute memory to
the correct service. It rejects ambiguous attribution or unrelated loaded models;
it does not unload somebody else's model to make the benchmark pass.

RSS sampling includes shared pages and can miss short spikes. Metal unified-memory
allocation is not fully represented by RSS. Ollama's reported model `size` and
`size_vram` are retained separately and must not be added to RSS as though they
were disjoint physical allocations. An actual 8GB device remains a separate test.

For the Mac benchmark, a dedicated instance of the installed Ollama runs on numeric
loopback port 11435, with cloud disabled, one request/model at a time and an OS
sandbox blocking all non-loopback networking. It shares downloaded local weights.
The Python GLM worker permits only its selected loopback port in its network audit
policy. CPU controls run with networking denied entirely. Both the dedicated
server and its model subprocesses inherit the OS sandbox. The instance is stopped
after measurement, while downloaded weights remain available for future local use.

Official references:
[GLM-OCR's Ollama guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md),
[model listing](https://ollama.com/library/glm-ocr:q8_0),
[Ollama generation API](https://docs.ollama.com/api/generate).
