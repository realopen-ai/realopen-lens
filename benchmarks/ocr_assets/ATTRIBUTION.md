# OCR diagnostic references

The manifest pins original images by SHA-256. Images are downloaded into ignored
local storage and are not bundled with the package.

## Terminal reference transcriptions

`terminal-*.txt` transcribe visible text from screenshots published by
[Textualize/Rich](https://github.com/Textualize/rich), revision
`9d8f9a372cc5916fd4781fec207ced7ddac2f08f`. Purely decorative box borders and
nontext prompt icons are omitted. Reading order is top to bottom, with source
file annotations placed at the end of their corresponding lines.

Original material is under the MIT License:

Copyright (c) 2020 Will McGugan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Diagram reference transcriptions

`diagram-*.txt` transcribe the example Internet Banking System diagrams by
[Simon Brown / C4 model](https://c4model.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They adapt the visual
layout into node text followed by relationship labels and the footer. That
chosen linear order is not a canonical reading order for a two-dimensional graph.

## Document references

Reference transcripts are fetched from
[tesseract-ocr/test](https://github.com/tesseract-ocr/test), revision
`232ff181c66516116ec0e84c4963f70de15050fd`, and retained locally with checksums.
The package does not redistribute the scanned articles or their transcripts.
The sample consists of two scanned article pages and one standard printed OCR
test image. See the upstream repository for individual test-data provenance.
