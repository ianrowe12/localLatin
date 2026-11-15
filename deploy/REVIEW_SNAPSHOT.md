# Encrypted review snapshots

Issue #285 freezes the review log and resolved CCL assignments for the second
paper dataset. It never writes to the production database or starts the app.

## Run and decrypt

Keep the RSA-4096 private key off GitHub. Set the PEM **public** key as the
repository variable `REVIEW_SNAPSHOT_PUBKEY_PEM`. The workflow also requires
`ENABLE_PRODUCTION_DEPLOY=true`, as does the account-role maintenance workflow.
`DEPLOY_PATH` defaults to `/homes/ipro222/localLatin`.

After the workflow is merged, an operator can dispatch it on `main`:

```bash
gh workflow run export-review-snapshot.yml --ref main
gh run list --workflow export-review-snapshot.yml --limit 1
gh run download RUN_ID --name encrypted-review-snapshot --dir snapshot-artifact
scripts/webapp/decrypt_review_snapshot.sh snapshot-artifact private-key.pem snapshot
```

Download within one day. The output directory must not already exist. Decryption
requires Python 3 and OpenSSL 1.1.1 or 3, with no Python packages. On macOS,
Apple's bundled LibreSSL may lack `enc -pbkdf2`; the helper refuses it rather
than falling back to a weaker KDF. Use Homebrew OpenSSL 3:

```bash
OPENSSL="$(brew --prefix openssl@3)/bin/openssl" \
  scripts/webapp/decrypt_review_snapshot.sh snapshot-artifact private-key.pem snapshot
```

The helper authenticates the ciphertext before decryption, rejects unexpected
archive paths and links, then verifies every CSV against `manifest.json`.
Nothing is published to the requested output directory until all checks pass.
Use encrypted local storage for the decrypted data.

## Contents and redaction

| File | Contents |
|---|---|
| `reviews.csv` | Every feedback row, ordered by insertion id: stored query id, witness xml_id, model, variant, outcome, ranks and saved directories, CCL key/action/directory/rank, schema version and review timestamp |
| `ccl_key_assignments.csv` | The API's resolved CCL table, using the same resolver and served rankings for legacy multi-rank picks, with pseudonyms instead of reviewer names |
| `reviewer_dirs.csv` | Directory id, seed query/witness, model, variant, CCL key, creation timestamp and pseudonymised creator |
| `reviewer_dir_members.csv` | Directory membership decisions, stored query/witness, membership role, timestamp and pseudonymised author |
| `manifest.json` | Snapshot schema version, UTC creation time, deployed git SHA, installed data release, webapp corpus digest, per-file row counts, outcome counts, distinct reviewed witnesses, feedback reviewer count, smoke/test feedback count and SHA256 of each CSV |

All three decision logs have `reviewer`, `reviewer_role` (current account role,
or `unknown` for legacy/deleted accounts) and `is_smoke_or_test`. The membership
`role` column is the seed/member decision, not the account role.

Reviewers become `R01`, `R02`, etc. in order of their first feedback **id**, not
timestamp. An account's renamed display names keep one pseudonym; accountless
historical rows use the API's stored-name identity rule. Directory-only authors
follow feedback authors, first by directory id, then membership id. This order
is snapshot-local, not an identifier to join across releases. SQL computes the
pseudonyms without persisting an identity lookup table or exporting a map.

Smoke/test rows are retained and flagged, not removed from CCL resolution.
Flags recognise the configured `LOCALLATIN_SMOKE_USERNAME` secret and whole
`smoke`, `test`, or `testing` tokens in usernames or current/historical display
names, including `locallatin-smoke` and `Deploy Smoke Check`. This is a
conservative heuristic for other test accounts, not proof that an unflagged
account is human. Inspect the flags before using the data in the paper.

No notes, directory display labels, usernames, display names, emails, account
ids, account audit records, passwords or sessions are exported. Decision-bearing
CCL keys and corpus/directory identifiers are retained as stored, not rewritten.
The SHA256 inventory covers every payload CSV; the manifest cannot hash itself.
The authenticated encryption envelope protects the manifest as well.

## Threat model and format

This repository is public. Treat **both Actions logs and artifacts as
world-readable**. Only counts and static failure messages are logged by the
script. The artifact contains exactly two files, ciphertext and an RSA-wrapped
random secret; neither the manifest nor any plaintext is uploaded.

The workflow shares the production-deploy concurrency lock, is main-only,
uses a fresh credential-free checkout and the production Python venv, and
opens the live database with SQLite `mode=ro`. The online backup API includes
committed WAL state consistently. All export processing uses temporary copies.
The temporary tree is private (umask 077) and removed on success or failure;
the workflow's `always()` cleanup also handles cancellation. A host crash or
forced kill can require local cleanup of that run's private staging directory.

A fresh 256-bit random secret, hex-encoded for OpenSSL's password-file input,
derives the AES-256-CBC key/IV through PBKDF2-HMAC-SHA256 with 200,000 iterations
and OpenSSL's random eight-byte salt. RSA-4096 OAEP wraps that secret with both
OAEP and MGF1 set to SHA256. A domain-separated SHA256 derivation of the secret
keys HMAC-SHA256 over the version header and encrypted bytes (encrypt-then-MAC).
`review-snapshot.tar.gz.enc` is `LLRS001\n`, 32 raw HMAC bytes, then OpenSSL's
salted ciphertext. `review-snapshot.key.enc` wraps the 64 hex digits plus newline.
No key material is passed on a command line.

CBC is intentional: OpenSSL `enc` does not support GCM. These options work on
RHEL 8's OpenSSL 1.1.1 and RHEL 9/macOS OpenSSL 3. Older RHEL 7 OpenSSL and some
LibreSSL versions lack PBKDF2 and are refused. See the upstream
[OpenSSL enc manual](https://docs.openssl.org/3.2/man1/openssl-enc/), including its
compatibility note about not using an explicit `-S` salt across OpenSSL versions.

Pseudonymisation is not anonymity: review timestamps, source identifiers and
decisions may still allow inference by someone with outside knowledge. The
private key holder, the production runner and reviewed workflow code are trusted.
Encryption does not protect a compromised runner or private key, and anyone
with the public key can fabricate a new bundle; this is not a digital signature
or an independently authenticated claim of origin.
