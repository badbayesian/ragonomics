# Streamlit UI

Run the local UI:

```bash
ragonometrics ui
```

Notes:
- The app includes Chat, DOI Network, and Usage tabs.
- Answers are concise and researcher-focused (with prompt engineering), with citations and snapshots.
- Optional page snapshots require `pdf2image` + Poppler and benefit from `pytesseract` for highlight overlays.
- External metadata (OpenAlex with CitEc fallback) is shown in an expander.
