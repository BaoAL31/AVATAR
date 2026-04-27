---
name: ingest-source-to-wiki
description: Ingest a new source document into the AVATAR wiki workflow by extracting takeaways, updating concept pages, linking related pages, and updating index/log files. Use when the user adds files to obsidian/raw and asks to ingest, summarize, or fold new evidence into the wiki.
---

# Ingest Source To Wiki

## Purpose

Use this skill when a user wants new source material integrated into the project wiki.

## Workflow

1. Read the full source document in `obsidian/raw/`.
2. Discuss key takeaways with the user before writing files.
3. Create a summary page in `obsidian/wiki/` named after the source.
4. Create or update concept pages for major ideas and entities.
5. Add `[[wiki-links]]` to connect related pages.
6. Update `obsidian/wiki/index.md` with new/updated pages.
7. Append an operation entry to `obsidian/wiki/log.md`.

## Hard Rules

- Never modify anything in `obsidian/raw/`.
- Every factual claim includes a citation in this format: `(source: filename.ext)`.
- If sources disagree, call out the contradiction explicitly.
- If a claim has no source, mark it as needing verification.
- Keep page names lowercase with hyphens.

## Page Template

Use this structure for all wiki pages:

```markdown
# Page Title

**Summary**: One to two sentences describing this page.

**Sources**: List of raw source files this page draws from.

**Last updated**: YYYY-MM-DD

---

Main content with clear headings and short paragraphs.

## Related pages

- [[related-page-1]]
- [[related-page-2]]
```

## Output Expectations

- Briefly report which pages were created vs updated.
- Call out unresolved ambiguities and missing evidence.
- Confirm `index.md` and `log.md` were updated.
