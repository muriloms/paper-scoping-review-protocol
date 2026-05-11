"""
REPLACEMENT / ADDITION CELLS FOR NOTEBOOK 01 — SECTION 2.4
============================================================
These cells document the keyword harmonization protocol.
Replace existing section 2.4 cells with these.
"""

# ========================================================================
# MARKDOWN CELL — 2.4 header (replace existing)
# ========================================================================
"""
### 2.4 Keyword Harmonization Protocol

**Problem:** Author keywords frequently contain morphological, orthographic, and lexical
variants of the same concept. If left unharmonized, these inflate the node count and
dilute the co-occurrence signal, producing a sparser, noisier network that compromises
reproducibility and analytical validity.

**Protocol:** A four-step controlled vocabulary harmonization was applied:

1. **Lowercase normalization:** All keywords were converted to lowercase to eliminate
   case-based duplicates (e.g., "Agent-Based Model" → "agent-based model").

2. **Whitespace and punctuation trimming:** Leading/trailing whitespace and inconsistent
   delimiters were standardized.

3. **Synonym merging via controlled vocabulary (thesaurus):** A manually curated lookup
   table maps known variants to canonical forms. The mapping rules fall into five categories:
   - *Singular/plural unification* (e.g., "agent-based models" → "agent-based modeling")
   - *Spelling variant normalization* (e.g., British "modelling" → American "modeling")
   - *Hyphenation normalization* (e.g., "agent based model" → "agent-based modeling")
   - *Abbreviation expansion* (e.g., "ABM", "ABMs" → "agent-based modeling")
   - *Scope unification* (e.g., "land use" → "land-use change")

4. **Post-merge deduplication:** After harmonization, duplicate keywords within the same
   article (arising from two variants mapping to the same canonical form) were removed.

The complete thesaurus is provided in **Supplementary Table S1**. Representative examples
are shown below.
"""

# ========================================================================
# CODE CELL — 2.4a Show harmonization map
# ========================================================================

from src.network_analysis import DEFAULT_KEYWORD_MAP

# Display the thesaurus grouped by rule category
rules = {
    "Synonym / Inflection": [],
    "Spelling variant (UK/US)": [],
    "Hyphenation normalization": [],
    "Abbreviation expansion": [],
    "Scope unification": [],
}

# Classify each mapping
for variant, canonical in sorted(DEFAULT_KEYWORD_MAP.items()):
    v_lower = variant.lower()
    if v_lower in ("abm", "abms", "mas", "agent-based model (abm)"):
        rules["Abbreviation expansion"].append((variant, canonical))
    elif "modelling" in v_lower:
        rules["Spelling variant (UK/US)"].append((variant, canonical))
    elif " " in v_lower and "-" not in v_lower and any(
        c in v_lower for c in ("agent", "multi", "land")
    ):
        rules["Hyphenation normalization"].append((variant, canonical))
    elif v_lower.endswith("s") or v_lower.endswith("es"):
        rules["Synonym / Inflection"].append((variant, canonical))
    else:
        rules["Scope unification"].append((variant, canonical))

print("KEYWORD HARMONIZATION THESAURUS")
print(f"Total mappings: {len(DEFAULT_KEYWORD_MAP)}")
print("=" * 65)
for category, pairs in rules.items():
    if pairs:
        print(f"\n{category}:")
        for variant, canonical in pairs:
            print(f"  '{variant}'  →  '{canonical}'")

# ========================================================================
# CODE CELL — 2.4b Before/after comparison
# ========================================================================

# --- RAW frequency (before harmonization) ---
raw_kws = df["Manual Tags"].dropna().str.split(";").explode().str.strip().str.lower()
raw_kws = raw_kws[raw_kws != ""]
raw_freq = raw_kws.value_counts()
n_raw = raw_freq.shape[0]

# --- Apply harmonization ---
df["kw_list"] = parse_keywords(df["Manual Tags"], sep=";")
df["n_keywords"] = df["kw_list"].apply(len)

all_keywords = [kw for kw_list in df["kw_list"] for kw in kw_list]
kw_freq = pd.Series(all_keywords).value_counts()
n_harmonized = kw_freq.shape[0]
n_merged = n_raw - n_harmonized

print("HARMONIZATION SUMMARY")
print("=" * 50)
print(f"  Unique keywords (raw):        {n_raw}")
print(f"  Unique keywords (harmonized): {n_harmonized}")
print(f"  Variants merged:              {n_merged}")
print(f"  Reduction:                    {n_merged/n_raw:.1%}")
print()

# Show the most impactful merges
print("MOST IMPACTFUL MERGES (by frequency gained):")
print("-" * 50)
for canonical in kw_freq.head(10).index:
    variants_merged = [v for v, c in DEFAULT_KEYWORD_MAP.items() if c == canonical]
    if variants_merged:
        raw_counts = {v: int(raw_freq.get(v, 0)) for v in variants_merged}
        raw_canonical = int(raw_freq.get(canonical, 0))
        total_after = int(kw_freq.get(canonical, 0))
        if sum(raw_counts.values()) > 0:
            print(f"\n  '{canonical}' (final count: {total_after})")
            if raw_canonical > 0:
                print(f"    original:  '{canonical}' = {raw_canonical}")
            for v, c in sorted(raw_counts.items(), key=lambda x: -x[1]):
                if c > 0:
                    print(f"    + merged:  '{v}' = {c}")

# ========================================================================
# CODE CELL — 2.4c Top 20 comparison table
# ========================================================================

# Side-by-side: top 20 before vs after
comparison = pd.DataFrame({
    "Before Harmonization": raw_freq.head(20),
    "After Harmonization": kw_freq.head(20),
})
print("\nTOP 20 KEYWORDS: BEFORE vs AFTER HARMONIZATION")
print("=" * 60)
print("\nBEFORE:")
print(raw_freq.head(20).to_string())
print("\nAFTER:")
print(kw_freq.head(20).to_string())

# ========================================================================
# CODE CELL — 2.4d Export thesaurus as supplementary table
# ========================================================================

# Export full thesaurus for Supplementary Material
thesaurus_rows = []
for variant, canonical in sorted(DEFAULT_KEYWORD_MAP.items()):
    # Determine rule category
    v = variant.lower()
    if v in ("abm", "abms", "mas", "agent-based model (abm)"):
        cat = "Abbreviation expansion"
    elif "modelling" in v:
        cat = "Spelling variant (UK/US)"
    elif " " in v and "-" not in v:
        cat = "Hyphenation normalization"
    elif v.endswith("s") and not v.endswith("ss"):
        cat = "Singular/Plural unification"
    else:
        cat = "Synonym merging"

    thesaurus_rows.append({
        "Variant": variant,
        "Canonical Form": canonical,
        "Rule Category": cat,
        "Raw Frequency": int(raw_freq.get(variant, 0)),
    })

thesaurus_df = pd.DataFrame(thesaurus_rows)
thesaurus_df = thesaurus_df.sort_values(["Rule Category", "Canonical Form", "Variant"])
thesaurus_path = os.path.join(FIG_DIR, "Table_S1_keyword_harmonization_thesaurus.csv")
thesaurus_df.to_csv(thesaurus_path, index=False)

print(f"Supplementary Table S1 exported to: {thesaurus_path}")
print(f"Total mappings: {len(thesaurus_df)}")
print(f"\nMappings per category:")
print(thesaurus_df["Rule Category"].value_counts().to_string())
