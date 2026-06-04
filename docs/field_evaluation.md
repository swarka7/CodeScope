# CodeScope Field Evaluation Notes

This document summarizes exploratory real-field validation runs on external Python projects. These runs are not a scientific benchmark and are not proof of production readiness. They are practical checks of whether CodeScope can point a developer toward useful files and functions outside the built-in benchmark apps.

## Purpose

The goal was to evaluate whether CodeScope generalizes to unfamiliar repositories and larger codebases by answering:

- Can CodeScope index the project successfully?
- Do `search` and `investigate` queries surface useful source files/functions?
- Do the top results help a developer decide what to inspect first?
- Where does retrieval still miss important counterpart logic?

## Methodology

For each project:

1. Build a local CodeScope index.
2. Run natural-language `investigate` queries and, where useful, semantic `search` queries.
3. Inspect the top 5 results.
4. Judge whether the results point to relevant files/functions and useful neighboring context.

Judgment criteria:

- **PASS**: main expected file/function appears in the top 3 or top 5 with clearly useful context.
- **PARTIAL**: relevant area appears, but an important counterpart area is missing or under-ranked.
- **FAIL**: top results are mostly unrelated.

## Summary

| project | indexed files | chunks | queries | result |
| --- | ---: | ---: | ---: | --- |
| File Server Protection System | 5 | 31 | 4 | PASS |
| Action Segmentation / Image Processing | 27 | 72 | 4 | 3 PASS, 1 PARTIAL |
| Port Scan Detector | 6 | 10 | 4 | 4 PASS, 0 PARTIAL, 0 FAIL |
| `psf/requests` scale test | external OSS repo | 704 | 3 | 2 PASS |

## File Server Protection System

Indexing result:

- Indexed files: 5
- Total chunks: 31

Queries tested:

- new file scan without blocking monitoring
- ClamAV infected file quarantine and logging
- startup scan plus continuous monitoring
- quarantine permissions logging search

Result: **PASS**

Useful retrieved areas included:

- `observer.py`
- `FileSecurityHandler`
- `_submit_scan`
- `on_created`
- `scan_file_worker`
- `scanner.py`
- `scan_file`
- `utils.py`
- `isolate_file`
- `create_secure_quarantine`
- `leave_warning_note`
- `logger.py`
- `log`

Notes:

- CodeScope surfaced both event-monitoring logic and scan/quarantine helpers.
- Retrieval covered the main workflow from file creation to scanning, quarantine, and logging.

## Action Segmentation / Image Processing

Indexing result:

- Indexed files: 27
- Total chunks: 72

Queries tested:

- feature/label shape mismatch
- duplicate labels to action segments
- annotation timestamps to frame labels
- target FPS / ResNet feature extraction

Result: **3 PASS, 1 PARTIAL**

Good retrieved areas included:

- `prepare_labels.py`
- `parse_annotations`
- `save_frame_labels`
- `parse_time`
- `run_inference.py`
- `inference_pipeline.py`
- `get_action_segments`
- `predict_from_npy.py`
- `compress_predictions`
- `extract_features.py`
- `extract_features_from_folder`
- `video_feature_utils.py`
- `extract_features_directly`
- `ms-tcn-master/makingNPY.py`
- `extract_features`

Weakness observed:

- The broad training shape/label mismatch query leaned too heavily toward feature extraction and did not surface enough dataset/training alignment code.

Notes:

- CodeScope performed well on concrete conversion, segmentation, timestamp, and feature-extraction questions.
- Multi-aspect queries that combine training data shape, labels, features, and model alignment still need better retrieval diversity.

## Port Scan Detector

Indexing result:

- Indexed files: 6
- Total chunks: 10

Queries tested:

- SYN packets trigger port scan alert
- only TCP SYN without ACK should be processed
- alert includes geoip/DNS/classification/ports/window
- stale SYN records removed before counting unique destination ports

Result: **4 PASS, 0 PARTIAL, 0 FAIL**

Useful retrieved areas included:

- `scanner_detector.py`
- `process_syn`
- `packet_sniffer.py`
- `syn_filter`
- `handle_pkt`
- `start_sniff`
- `alert_manager.py`
- `alert`
- `classify_scan`
- `reverse_dns`
- `geoip_lookup`
- `utils.py`
- `current_time`

Notes:

- CodeScope strongly surfaced the central packet-processing and alerting flow.
- Related context helped connect alert generation to DNS, GeoIP, classification, and time-window helpers.

## `psf/requests` Scale Test

This run tested CodeScope on a real open-source Python repository that is larger than the built-in examples.

Indexing and timing:

- Total chunks: 704
- `.codescope` index size: about 9.07 MB
- Initial index time: about 120.3 seconds
- Search time: about 11.2 seconds
- Investigate redirect query time: about 10.6 seconds
- Investigate cookie/session query time: about 10.6 seconds
- JSON investigate time: about 13.3 seconds
- JSON output size: about 11.7 KB

Queries tested:

- search: `session request authentication headers cookies redirect`
- investigate: redirects should preserve/remove authentication headers depending on target host
- investigate: session cookies should persist across requests and redirects

Result:

- Redirect/auth query: **PASS**
- Cookie/session query: **PASS**, with a note that multi-aspect diversity could improve

Good retrieved areas included:

- `requests/sessions.py`
- `SessionRedirectMixin.resolve_redirects`
- `SessionRedirectMixin.get_redirect_target`
- `SessionRedirectMixin.should_strip_auth`
- `SessionRedirectMixin.rebuild_auth`
- `HTTPDigestAuth.handle_redirect`
- `RequestsCookieJar`
- `RequestsCookieJar.set`

Weakness observed:

- For the cookie/session query, CodeScope found relevant session, redirect, and cookie classes, but could better diversify toward `merge_cookies`, `extract_cookies_to_jar`, `Session.prepare_request`, or `Session.send`.

Notes:

- The redirect/auth behavior query showed strong retrieval of the correct session redirect logic.
- The cookie/session query still found useful areas, but the top results could spread better across cookie merge/extract and session send/prepare code paths.
- Performance was usable for exploratory CLI work, but model startup overhead was visible on every command.

## Limitations

- These are exploratory field notes, not a controlled scientific benchmark.
- Small external repos are not proof of production readiness.
- `psf/requests` is larger than the built-in examples but still not a huge monorepo.
- Timings came from one Windows development machine.
- Hugging Face model loading overhead affects each command.
- CodeScope is Python-focused.
- Retrieval rankings are heuristic and can miss important counterpart logic in broad multi-aspect queries.

## Takeaways

- CodeScope generalized beyond the built-in benchmark apps.
- Results were especially strong on the Port Scan Detector and `psf/requests` redirect/auth behavior.
- The main improvement area is multi-aspect query diversity: broad queries should better distribute results across related subsystems instead of over-concentrating on one aspect.
- Performance can improve by avoiding repeated embedding model startup on every CLI command.

## Future Improvements

- Reuse an embedding model process/server to avoid reloading embeddings every command.
- Test against larger repositories.
- Expand the benchmark set with more domains and failure modes.
- Improve multi-aspect retrieval coverage.
- Explore optional local embedding optimization or ONNX-based inference later.
