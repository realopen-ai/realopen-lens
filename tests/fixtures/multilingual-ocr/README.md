# Korean OCR regression

`korean-document.png` is the deterministic `kor-0` synthetic diagnostic from
`benchmarks.datasets.multilingual` (manifest and pinned Google Fonts provenance in
`benchmarks/sources/multilingual-ocr-v1/`). Text and generator: Apache-2.0.
The Noto Sans KR rendering font is SIL-OFL-1.1; font files are not included here.

Expected text:

```text
청구서 번호 47291
총 금액 125 유로
오늘 결제를 받았습니다
```

Tesseract 5.5.2 with the pinned fast Korean pack fragments this layout in automatic
and sparse modes. The bounded block-layout retry recovers the main text. This is a
regression sample, not independent evidence of general Korean OCR accuracy.
