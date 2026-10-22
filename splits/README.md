# Frozen splits

`synthrad2025_local_test.json` and its `.sha256` are created once from the real SynthRAD2025 data with
`sctrange-split` and committed. They are never regenerated: `sctrange.split.load_split` fails if the file
no longer matches its recorded hash, and the CLI refuses to overwrite an existing split.

Test cases are opened a single time, for the final evaluation.
