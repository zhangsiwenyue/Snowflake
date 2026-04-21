"""High-level helpers that summarise the Census schema for the LLM.

The Cybersyn share contains ~70 wide tables with cryptic codes (B01001e1, ...).
Two patterns we expose to the agent:

  1. `list_table_families()` — a static, hand-curated map of which ACS table
     family lives in which physical table (e.g. B19 = income → 2020_CBG_B19).
     This goes in the system prompt so the model can pick the right table
     without making a tool call for trivial questions.

  2. `search_fields(keyword, year)` — full-text-ish search over the metadata
     table FIELD_LEVEL_* columns. The model calls this whenever it needs to
     translate a human concept ("median household income") to a column code.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .snowflake_client import SnowflakeClient

# Curated, human-written description of the ACS table families.
# Source: https://www.census.gov/programs-surveys/acs/technical-documentation/table-shells.html
# We keep this static (and small) so the system prompt stays under 2k tokens.
TABLE_FAMILIES: dict[str, str] = {
    "B01": "Sex by Age, Median Age, Total Population (use B01001 for age/sex breakdowns, B01003 for total population)",
    "B02": "Race (single-race population counts)",
    "B03": "Hispanic or Latino origin",
    "B07": "Geographic mobility (residence 1 year ago)",
    "B08": "Means of transportation to work, commute time",
    "B09": "Children, household relationships",
    "B11": "Households and families (household type, size)",
    "B12": "Marital status",
    "B14": "School enrollment",
    "B15": "Educational attainment",
    "B16": "Language spoken at home, English-speaking ability",
    "B17": "Poverty status",
    "B19": "Income (B19013 = median household income, B19001 = household income brackets, B19301 = per-capita income)",
    "B20": "Earnings",
    "B21": "Veteran status",
    "B22": "Food stamps / SNAP receipt",
    "B23": "Employment status, labor force",
    "B24": "Industry and occupation",
    "B25": "Housing (B25001=units, B25003=tenure, B25064=median gross rent, B25077=median home value)",
    "B27": "Health insurance coverage",
    "B28": "Computer and internet use",
    "B29": "Citizen voting-age population",
    "B99": "Imputation flags",
    "C02": "Race (collapsed categories)",
    "C15": "Educational attainment (collapsed)",
    "C16": "Language spoken at home (collapsed)",
    "C17": "Poverty (collapsed)",
    "C21": "Veteran status (collapsed)",
    "C24": "Industry/occupation (collapsed)",
}

YEARS_AVAILABLE = (2019, 2020)


@dataclass(frozen=True)
class FieldHit:
    column_id: str       # e.g. "B19013e1"
    table_id: str        # e.g. "B19013"
    table_title: str     # e.g. "Median Household Income"
    description: str     # joined hierarchy, e.g. "Estimate > Median household income"
    physical_table: str  # e.g. "2020_CBG_B19"


def _join_levels(row: dict) -> str:
    # The underlying column 9 is mis-cased as "FIELD_LEVELl_9" in the share;
    # we alias it to FIELD_LEVEL_9 in our SELECT so the Python side is uniform.
    parts = [row.get(f"FIELD_LEVEL_{i}") for i in range(1, 11)]
    return " > ".join(p for p in parts if p)


class CensusCatalog:
    def __init__(self, client: SnowflakeClient):
        self.client = client

    def search_fields(
        self, keyword: str, year: int = 2020, limit: int = 25
    ) -> list[FieldHit]:
        """Search the metadata table for fields whose hierarchy contains all
        whitespace-separated keywords. Case-insensitive AND across tokens."""
        if year not in YEARS_AVAILABLE:
            raise ValueError(f"year must be one of {YEARS_AVAILABLE}, got {year}")
        tokens = [t for t in keyword.strip().split() if t]
        if not tokens:
            return []

        # Build a WHERE clause that ANDs ILIKE on each token across the whole row.
        # We concat all hierarchy fields into a single searchable string per row.
        searchable = (
            "COALESCE(TABLE_TITLE,'')||' '||COALESCE(TABLE_TOPICS,'')||' '||"
            "COALESCE(TABLE_UNIVERSE,'')||' '||COALESCE(FIELD_LEVEL_1,'')||' '||"
            "COALESCE(FIELD_LEVEL_2,'')||' '||COALESCE(FIELD_LEVEL_3,'')||' '||"
            "COALESCE(FIELD_LEVEL_4,'')||' '||COALESCE(FIELD_LEVEL_5,'')||' '||"
            "COALESCE(FIELD_LEVEL_6,'')||' '||COALESCE(FIELD_LEVEL_7,'')||' '||"
            'COALESCE(FIELD_LEVEL_8,\'\')||\' \'||COALESCE("FIELD_LEVELl_9",\'\')||\' \'||'
            "COALESCE(FIELD_LEVEL_10,'')"
        )
        where = " AND ".join([f"{searchable} ILIKE ?" for _ in tokens])
        params = [f"%{t}%" for t in tokens]
        # Prefer estimate columns over margin-of-error columns by default.
        sql = f"""
            SELECT TABLE_ID, TABLE_NUMBER, TABLE_TITLE, TABLE_TOPICS, TABLE_UNIVERSE,
                   FIELD_LEVEL_1, FIELD_LEVEL_2, FIELD_LEVEL_3, FIELD_LEVEL_4,
                   FIELD_LEVEL_5, FIELD_LEVEL_6, FIELD_LEVEL_7, FIELD_LEVEL_8,
                   "FIELD_LEVELl_9" AS FIELD_LEVEL_9, FIELD_LEVEL_10
              FROM "{year}_METADATA_CBG_FIELD_DESCRIPTIONS"
             WHERE {where}
               AND TABLE_ID NOT ILIKE '%m_'
               AND TABLE_ID NOT ILIKE '%m__'
             ORDER BY LENGTH(TABLE_ID), TABLE_ID
             LIMIT {int(limit)}
        """
        rows, _ = self.client.execute(sql, params)
        hits: list[FieldHit] = []
        for r in rows:
            col = r["TABLE_ID"]
            family = (r.get("TABLE_NUMBER") or "")[:3]
            physical = f"{year}_CBG_{family}" if family else ""
            hits.append(
                FieldHit(
                    column_id=col,
                    table_id=r.get("TABLE_NUMBER") or "",
                    table_title=r.get("TABLE_TITLE") or "",
                    description=_join_levels(r),
                    physical_table=physical,
                )
            )
        return hits

    @lru_cache(maxsize=1)
    def state_fips_map(self) -> dict[str, str]:
        """Return {STATE_ABBR: STATE_FIPS} for all 50 states + DC + territories."""
        sql = (
            'SELECT DISTINCT STATE, STATE_FIPS '
            'FROM "2020_METADATA_CBG_FIPS_CODES" '
            'ORDER BY STATE'
        )
        rows, _ = self.client.execute(sql)
        return {r["STATE"]: r["STATE_FIPS"] for r in rows}
