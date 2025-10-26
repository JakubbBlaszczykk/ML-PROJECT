# clean_imdb_duckdb_final_safe.py
import duckdb
import math

# CONFIG
INPUT_URL = "https://workspace4824871889.blob.core.windows.net/azureml-blobstore-84f516da-0fe5-4f33-8f3c-f18ec8e2b4f7/UI/2025-10-22_105430_UTC/imdb_merged_duckdb.parquet"  # can be local path or https URL (if accessible)   # local path or http(s) URL if accessible
OUTPUT = "imdb_merged_cleaned_duckdb.parquet"
DROP_COLS = {"ordering_x", "ordering_y", "attributes", "job", "ordering_akas", "ordering_principal"}
CATEGORICAL_FILL = "unknown"
NUMERIC_IMPUTE_COLS = ['averageRating', 'numVotes', 'seasonNumber', 'episodeNumber']
TRANSFORM_STRING_COLS = ['primaryTitle', 'primaryName', 'title', 'region', 'language']
LIST_COLS = ['genres', 'directors', 'writers']
CHAR_COL = 'characters'
YEAR_COLS = ['startYear', 'birthYear', 'deathYear']

# Helper: classifies DuckDB types simply
def is_numeric_type(duck_type: str):
    t = duck_type.lower()
    return any(x in t for x in ["tinyint","smallint","integer","int","bigint","decimal","numeric","float","double"])

def is_int_type(duck_type: str):
    t = duck_type.lower()
    return any(x in t for x in ["tinyint","smallint","integer","int","bigint"])

def is_string_type(duck_type: str):
    t = duck_type.lower()
    return any(x in t for x in ["varchar","text","string","char"])

con = duckdb.connect()

# 1) Register parquet as a view
con.execute(f"CREATE OR REPLACE VIEW imdb_raw AS SELECT * FROM read_parquet('{INPUT_URL}');")

# 2) Get actual column names and types
cols_info = con.execute("DESCRIBE imdb_raw").fetchall()  # returns list of (name, type, null?)
all_cols = [(r[0], r[1]) for r in cols_info]

# 3) compute medians for imputeable numeric columns using TRY_CAST
medians = {}
for col, coltype in all_cols:
    if col in NUMERIC_IMPUTE_COLS:
        if is_numeric_type(coltype):
            try:
                res = con.execute(
                    f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY TRY_CAST({col} AS DOUBLE)) FROM imdb_raw WHERE {col} IS NOT NULL AND {col} != '\\\\N'"
                ).fetchone()
                med = res[0] if res and res[0] is not None else 0
            except Exception:
                med = 0
        else:
            # if the declared type is string, try to compute median by casting values
            try:
                res = con.execute(
                    f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY TRY_CAST({col} AS DOUBLE)) FROM imdb_raw WHERE TRY_CAST({col} AS DOUBLE) IS NOT NULL"
                ).fetchone()
                med = res[0] if res and res[0] is not None else 0
            except Exception:
                med = 0
        if med is None or (isinstance(med, float) and (math.isnan(med) or math.isinf(med))):
            med = 0
        medians[col] = med

# 4) compute numVotes quantile bounds for outlier filtering using TRY_CAST
low_q, high_q = 0, 10**12
if any(c == 'numVotes' for c, _ in all_cols):
    try:
        low, high = con.execute(
            "SELECT percentile_cont(0.01) WITHIN GROUP (ORDER BY TRY_CAST(numVotes AS DOUBLE)), percentile_cont(0.999) WITHIN GROUP (ORDER BY TRY_CAST(numVotes AS DOUBLE)) FROM imdb_raw WHERE TRY_CAST(numVotes AS DOUBLE) IS NOT NULL"
        ).fetchone()
        if low is not None:
            low_q = int(max(0, math.floor(low)))
        if high is not None:
            high_q = int(math.ceil(high))
    except Exception:
        pass

# 5) Build safe SELECT expressions per column (explicit)
select_parts = []

for col, coltype in all_cols:
    if col in DROP_COLS:
        continue

    # prioritize explicit higher-level transforms
    if col in NUMERIC_IMPUTE_COLS:
        med = medians.get(col, 0)
        # Use TRY_CAST to produce numeric double, fallback to median
        select_parts.append(f"COALESCE(TRY_CAST({col} AS DOUBLE), {med}) AS {col}")
        continue

    if col in YEAR_COLS:
        # attempt integer cast, else NULL
        # If underlying type is numeric, just TRY_CAST to BIGINT; otherwise TRY_CAST string to BIGINT
        select_parts.append(f"TRY_CAST({col} AS BIGINT) AS {col}")
        continue

    if col in TRANSFORM_STRING_COLS:
        # string normalization: replace '\N' or NULL -> 'unknown', else lower(trim(...))
        # But if underlying declared type is numeric, be safe: TRY_CAST -> if numeric then cast to string via CAST(... AS VARCHAR)
        if is_string_type(coltype):
            select_parts.append(f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '{CATEGORICAL_FILL}' ELSE lower(trim({col})) END AS {col}")
        else:
            # declared numeric but you still want to present as string: convert safely
            select_parts.append(f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '{CATEGORICAL_FILL}' WHEN TRY_CAST({col} AS DOUBLE) IS NOT NULL THEN lower(trim(CAST(TRY_CAST({col} AS DOUBLE) AS VARCHAR))) ELSE '{CATEGORICAL_FILL}' END AS {col}")
        continue

    if col in LIST_COLS:
        # normalize commas and lowercase; treat '\N' as unknown
        if is_string_type(coltype):
            select_parts.append(
                f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '{CATEGORICAL_FILL}' ELSE lower(regexp_replace(regexp_replace({col}, '\\\\s*,\\\\s*', ',', 'g'), '^,+|,+$', '', 'g')) END AS {col}"
            )
        else:
            select_parts.append(
                f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '{CATEGORICAL_FILL}' WHEN TRY_CAST({col} AS DOUBLE) IS NOT NULL THEN lower(regexp_replace(regexp_replace(CAST(TRY_CAST({col} AS DOUBLE) AS VARCHAR), '\\\\s*,\\\\s*', ',', 'g'), '^,+|,+$', '', 'g')) ELSE '{CATEGORICAL_FILL}' END AS {col}"
            )
        continue

    if col == CHAR_COL:
        if is_string_type(coltype):
            select_parts.append(
                f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '' ELSE lower(regexp_replace(regexp_replace(regexp_replace({col}, '\\\\[|\\\\]|\"', '', 'g'), '\\\\s*,\\\\s*', ',', 'g'), '^,+|,+$', '', 'g')) END AS {col}"
            )
        else:
            select_parts.append(
                f"CASE WHEN {col} IS NULL OR {col} = '\\\\N' THEN '' WHEN TRY_CAST({col} AS DOUBLE) IS NOT NULL THEN lower(regexp_replace(regexp_replace(regexp_replace(CAST(TRY_CAST({col} AS DOUBLE) AS VARCHAR), '\\\\[|\\\\]|\"', '', 'g'), '\\\\s*,\\\\s*', ',', 'g'), '^,+|,+$', '', 'g')) ELSE '' END AS {col}"
            )
        continue

    # Default handling based on declared type:
    if is_numeric_type(coltype):
        # Ensure numeric output: TRY_CAST to DOUBLE (NULL if not castable)
        select_parts.append(f"TRY_CAST({col} AS DOUBLE) AS {col}")
    elif is_string_type(coltype):
        # Replace literal '\N' with NULL; keep string as-is (optionally trim/lower if you want)
        select_parts.append(f"CASE WHEN {col} = '\\\\N' THEN NULL ELSE {col} END AS {col}")
    else:
        # fallback: try to TRY_CAST to DOUBLE, else pass through with '\N' -> NULL guard
        select_parts.append(f"CASE WHEN {col} = '\\\\N' THEN NULL WHEN TRY_CAST({col} AS DOUBLE) IS NOT NULL THEN TRY_CAST({col} AS DOUBLE) ELSE {col} END AS {col}")

# 6) Assemble final SQL
final_select_sql = ",\n    ".join(select_parts)
where_clause = "1=1"
if any(c == 'numVotes' for c, _ in all_cols):
    # use TRY_CAST in where to avoid conversion errors
    where_clause = f"TRY_CAST(numVotes AS DOUBLE) BETWEEN {low_q} AND {high_q}"

create_view_sql = f"""
CREATE OR REPLACE VIEW imdb_cleaned_final AS
SELECT
    {final_select_sql}
FROM imdb_raw
WHERE {where_clause}
;
"""

# Optional: print a snippet for debugging
# print(create_view_sql[:2000])

# 7) Execute view creation & export
con.execute(create_view_sql)

print(f"Writing cleaned parquet to: {OUTPUT}")
# use COPY which streams
con.execute(f"COPY (SELECT * FROM imdb_cleaned_final) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');")
print("Done. Output:", OUTPUT)

con.close()
