# Model evidence for amendment 5

The release dates, licence and OpenRouter listing that [protocol.json](../protocol.json)
(`arms.llm.weights`, `contamination`, amendment 5) cites for the LLM arm's model. All
three files are public API responses, committed unchanged as fetched at
**2026-10-04T20:18:59Z** (2026-10-05 04:18:59 UTC+08:00). No LLM was called to get them.

| File | URL | What it shows | SHA-256 |
|---|---|---|---|
| [hf-qwen38-rev.json](hf-qwen38-rev.json) | `https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | The pinned revision exists: `sha` is the pinned revision, `lastModified` 2026-08-14T15:00:01Z, `createdAt` 2026-08-05T08:22:59Z (the repository's creation), `license:apache-2.0` in `tags` and `cardData` | `888ca751f3379f08cd73b4a6025243c0e73dbcad828fe9691ab30b6c142880d8` |
| [hf-qwen38-commits.json](hf-qwen38-commits.json) | `https://huggingface.co/api/models/Qwen/Qwen3.8-27B/commits/main` | The commit history of `main`: the pinned revision is the commit of 2026-08-14T15:00:01Z ("Update README.md"); the files were uploaded in earlier commits (e.g. "Upload folder using huggingface_hub", 2026-08-13); the initial commit is 2026-08-05T08:22:59Z | `89ce4c5ed75bf376ffc68dc1ede542db4be166d33b92dce7f9aae6039de8330e` |
| [openrouter-qwen38-free.json](openrouter-qwen38-free.json) | `https://openrouter.ai/api/v1/models` (the entry whose `id` is `qwen/qwen3.8-27b:free`, extracted from the list) | `created` 1786722910 (2026-08-14 15:55:10 UTC), `hugging_face_id` `Qwen/Qwen3.8-27B`, prompt and completion price 0 | `2b6ef2b56cdb7bc8570bd6f2c13a57757ed8faf793d8cdd091466798e143e893` |

**What follows.** The pinned weights were published (2026-08-14 at the latest) before
the test window starts (2026-09-01), so before any test-window bar existed; those
weights cannot have been trained on test-window bars.

**What does not.** OpenRouter does not say which revision or quantisation it serves, and
the arm's `model_check` compares only the response's model id string. That the arm runs
these weights rests on OpenRouter serving this release.
