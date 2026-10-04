# Historical source manifest

`qrae.historical-source-manifest.v1` is the local intake contract for historical `qrae.price-bars.v1` data. It records provenance and verifies exact files; it is not a legal opinion, a downloader, or E1 evidence by itself.

## Canonical families

- `PREDICTION_MARKETS`
- `CRYPTO`
- `EQUITIES_ETFS`
- `FUTURES_RATES_COMMODITIES_FX`
- `OPTIONS`
- `ALTERNATIVE_SOCIAL_WALLET`

Each family requires the exact control names enforced by `MARKET_FAMILY_CONTROLS` in `src/qrae/historical_data.py`. Each control points to a local regular file with SHA-256 and byte size. Evidence files must contain no credentials.

## Required objects

The strict manifest contains:

- dataset/provider IDs and a credential-free HTTPS `source_uri`;
- a `price_data` file reference with schema `qrae.price-bars.v1`;
- declared `LOCAL_RESEARCH_ONLY` entitlement interval, hashed entitlement ID, evidence type, and evidence file;
- source publication, acquisition, and ingestion timestamps;
- IANA timezone, session policy, and calendar evidence;
- immutable/append-only/vintaged/restatable revision policy and evidence;
- the exact family-control evidence map.

Start from `packages/research/examples/historical_source/manifest.template.json`. It is intentionally invalid until every placeholder hash and byte size is replaced; its companion README shows local hashing commands.

Every file reference has only `path`, `sha256`, and `size_bytes`. Paths are normalized workspace-relative POSIX paths. HTTPS queries are allowed only when their names/content are not credential-like.

`price_data` has exactly:

```text
event_time,available_time,instrument,price
```

In this schema `event_time` is the decision timestamp and `available_time` must be no later. Decision timestamps must be unique and strictly increasing per instrument, not exceed `as_of`, and carry finite positive prices. A provider vintage must first be materialized as one as-of decision series; duplicate or non-monotonic instrument/decision timestamps are rejected.

## Import and evidence ceiling

```text
python -m qrae.cli historical-import \
  --catalog-root .quantos/catalog \
  --workspace . \
  --manifest data/historical-manifest.json \
  --as-of 2025-01-01T00:00:00Z \
  --request-id history-001
```

The catalog stores a canonical `qrae.historical-import-bundle.v1` raw object and the verified CSV as the normalized object. The bundle preserves the exact manifest text/hash, canonical manifest hash, validation record/hash, evidence references, validation cutoff/time, and validated revision parent. Offline verification requires both paired objects: it rehashes the normalized CSV and recomputes its row, instrument, and temporal quality fields rather than trusting a rehashed validation record. This proves internal consistency, not authenticity by itself: rely on the trusted catalog snapshot/object hashes (or a separately authenticated receipt); an actor able to replace both objects and every expected hash can forge a coherent standalone pair. The adapter is `LOCAL_MANIFEST_IMPORT`/`manifest-declared`; registration remains `TIER_0`, `OBSERVED_AT_IMPORT`, and `E0`.

The first cataloged vintage must declare no parent. A later vintage must name the canonical hash of the latest snapshot in the same dataset/provider lineage. Mode changes, duplicate vintage IDs, missing parents, forks, cycles through a stale parent, and publication/acquisition/ingestion or validation-time regression fail closed. Standalone validation is never data-input-eligible because it cannot prove catalog lineage; only this catalog check can set that mechanical flag.

The catalog preserves the declared expiry but does not yet re-run this manifest validator when a later work order resolves the snapshot. Revalidate the manifest before each new research use. Work-order-time entitlement revalidation is a required future gate before any E1 unlock.

`e1_eligibility.data_input_eligible=true` means only that mechanical manifest, file, PIT, expiry, monotonicity, family-control, and required catalog-lineage checks passed. `e1_claim_authorized` and `legal_license_truth_verified` remain false. E1 additionally requires human/provider entitlement review and compatible chronological temporal validation with costs.
