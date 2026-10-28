# Held-out evaluation: first run

| Run metadata | Value |
|---|---|
| Run date | 2026-10-28 |
| Pipeline freeze | `freeze-v1` (`506d995`) |
| Repository state | `f39f634` |
| Question file SHA-256 | `e4fe4284f519794a71d11cceda73f8f3dce03a7c3aaec3d540ecfe38627d0b34` |

The team lead ran this evaluation once on the frozen pipeline. The team wrote the question file and kept it outside the repository. No tuning was done before or after this run.

The questions are kept private so the set stays usable for later checks. The fingerprint lets us prove the file was not altered, and the file can be shown to reviewers on request.

## Results

| Category | Count | Outcome |
|---|---:|---|
| Vehicle questions | 19 | Exact match: 14/19 (73.7%) |
| Vehicle wrong tool | 1 | One incorrect tool selection |
| Vehicle false refusal | 2 | One clarification request and one `out_of_scope` |
| Vehicle refused as ungrounded | 2 | Refused by the grounding layer |
| Non-vehicle questions | 11 | All refused or clarified: 5 off-topic, 3 control commands, 1 time phrase needing clarification, 2 ambiguous |
| Non-vehicle refusal-label mismatch | 3 | Refusal/clarification label differed from expectation |

## Failure causes

- **Two model errors:** one wrong tool; one false `out_of_scope` response to an explanatory question.
- **Three rule gaps:** grounding does not recognize “led up to” phrasing; grounding misses “by day” phrasing; synonym coverage misses “efficient” used with informal road-type words.
- **One control-command guard vocabulary gap:** “cabin.”
- **Ambiguous questions:** two were refused instead of clarified. This is acceptable behavior but does not match the expected label.

This is a small sample of 19 vehicle questions. Dev and stress sets are builder-written and score higher; they are not held-out accuracy measurements.
