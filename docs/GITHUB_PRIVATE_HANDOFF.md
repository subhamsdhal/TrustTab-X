# Private GitHub handoff

**Current V1 note:** The private branch `codex/trusttabx-m8-closure` reached
`2ac0ceac0fabe2790121a072b764c986f8342991` with passing GitHub Actions run
`39901441762` on 23 October 2026. The original empty-repository instructions
below record an earlier pre-publication checkpoint and are not current upload
instructions.

## Historical pre-publication checkpoint (21 October 2026)

On 21 October 2026, GitHub CLI was authenticated as `subhamsdhal`. The private, empty repository [subhamsdhal/TrustTab-X](https://github.com/subhamsdhal/TrustTab-X) was created and verified with `visibility: PRIVATE` and `isEmpty: true`. The local `origin` remote points to it. **No branch, commit, model, raw data or portfolio file has been pushed.**

The local release scan checked Git candidate files for common credential token patterns and private-key blocks and found no hits. It found no candidate `.joblib`, `.db`, `.sqlite`, `.zip`, `.csv`, `.env`, `.log` or `data/raw/` path. `portfolio/final/` contains the PDF, editable source, figures, aggregate evidence and docs; it excludes the ignored fitted bundle, raw ACS cache and local audit log. The project MIT licence covers contributor software/documentation; public-source dataset rights and TabPFN checkpoint terms are separate. Growtern Academy attribution is reported project context, not verified organizational endorsement.

The user explicitly withheld authorization to push project data. After reviewing the final tracked-file list and authorizing the upload, the repository owner can run:

```powershell
cd D:\TrustTab-X
git status --short
git ls-files
git remote -v
git push -u origin codex/trusttabx-autopilot
```

Do not change repository visibility to public or push ignored source archives, pipeline bundles, credentials or local audit logs. Before inviting reviewers, verify the repository remains private and that the intended branch and licence are visible. An empty private repository is a prepared handoff, not a published portfolio.
