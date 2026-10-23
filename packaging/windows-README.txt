RealOpen Lens by RealOpen AI - Windows x64 (experimental)

Extract the complete ZIP into a writable local folder before starting
RealOpenLens.exe. Keep _internal beside the executable. Do not run from inside
the ZIP. Python, the gallery and Tesseract (English/French/Arabic/orientation)
are included; Python, Node.js and a separate OCR install are not needed.

Device setup offers an explicit model download on first launch. Internet is
needed for that download; indexing and searching then work offline. INT8 is
suggested below 16 GB RAM, FP32 at 16 GB or above; E5 stays INT8.

Open image library, choose Add folder, and start indexing. Original files stay
in place. Use Quit Lens to stop the local gallery. Your library and logs live
under %LOCALAPPDATA%\RealOpenLens (independent of this extracted directory).

This unsigned experimental application may trigger Windows security prompts.
Only run a download from the project release you trust. Do not disable antivirus.

License and dependency notices: _internal\licenses\realopen-lens\
OCR dependency versions/licenses: _internal\lens-ocr\
Source and documentation: https://github.com/realopen-ai/realopen-lens
