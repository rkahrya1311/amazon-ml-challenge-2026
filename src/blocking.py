"""Generate and validate candidate business pairs from B1, B2, and B3."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SOURCE_FILES = {
    "S1": "train_source1.tsv",
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"
TEST_SOURCE_FILES = {
    "S1": "test_source1.tsv",
    "S2": "test_source2.tsv",
    "S3": "test_source3.tsv",
}

NAME_STOP_WORDS = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd",
    "limited", "co", "company", "plc", "gmbh", "the", "and",
}
ADDRESS_STOP_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr",
    "suite", "ste", "unit", "floor", "fl", "building", "bldg",
    "block", "near", "opp", "opposite", "behind", "plot", "no",
    "number", "dist", "district",
}

OUTPUT_COLUMNS = ["s1_id", "matched_id", "matched_source", "blocking_rule"]


def sql_path(path: Path) -> str:
    """Return a single-quoted SQL path with embedded quotes escaped."""
    return "'" + path.as_posix().replace("'", "''") + "'"


def read_tsv_sql(path: Path) -> str:
    """Build the DuckDB reader used by the original tab-separated inputs."""
    return (
        f"read_csv({sql_path(path)}, delim='\\t', header=true, "
        "all_varchar=true, quote='', strict_mode=false, null_padding=true)"
    )


def clean_input_files(
    data_dir: Path,
    temporary_dir: Path,
    source_files: dict[str, str],
    ground_truth_file: str | None,
) -> dict[str, Path]:
    """Copy input lines to UTF-8, replacing malformed bytes without dropping rows."""
    cleaned_dir = temporary_dir / "cleaned_inputs"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    required_names = list(source_files.values())
    if ground_truth_file is not None:
        required_names.append(ground_truth_file)
    cleaned_paths: dict[str, Path] = {}

    for file_name in required_names:
        source_path = data_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {source_path}")

        cleaned_path = cleaned_dir / file_name
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with source_path.open("rb") as source, cleaned_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            while chunk := source.read(8 * 1024 * 1024):
                destination.write(decoder.decode(chunk))
            destination.write(decoder.decode(b"", final=True))
        cleaned_paths[file_name] = cleaned_path

    return cleaned_paths


def sql_string_list(values: set[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in sorted(values))


def normalized_text_sql(column_name: str) -> str:
    """Lowercase text and replace non-letter/digit runs with spaces."""
    return (
        "regexp_replace("
        f"lower(coalesce({column_name}, '')), "
        "'[^\\p{L}\\p{N}]+', ' ', 'g')"
    )


def create_source_views(
    connection: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
    source_files: dict[str, str],
) -> None:
    for source, file_name in source_files.items():
        view_name = f"training_{source.lower()}"
        connection.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM {read_tsv_sql(paths[file_name])}"
        )


def create_ground_truth(connection: duckdb.DuckDBPyConnection, path: Path) -> None:
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE ground_truth AS
        SELECT DISTINCT
            rows.source1_entity_id AS s1_id,
            trim(matches.matched_id) AS matched_id
        FROM {read_tsv_sql(path)} AS rows,
        UNNEST(string_split(rows.matched_entity_ids, ','))
            AS matches(matched_id)
        WHERE trim(matches.matched_id) <> ''
    """)


def create_target_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE training_targets AS
        SELECT entity_id, business_name, business_address, country
        FROM training_s2
        UNION ALL
        SELECT entity_id, business_name, business_address, country
        FROM training_s3
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE target_source_ids AS
        SELECT DISTINCT entity_id AS matched_id, 'S2' AS matched_source
        FROM training_s2
        UNION ALL
        SELECT DISTINCT entity_id AS matched_id, 'S3' AS matched_source
        FROM training_s3
    """)


def rule_key_queries(blocking_rule: str) -> tuple[str, str]:
    """Return target and S1 key queries without changing B1, B2, or B3."""
    if blocking_rule == "B1":
        stop_words = sql_string_list(NAME_STOP_WORDS)
        target_name = normalized_text_sql("business_name")
        s1_name = normalized_text_sql("business_name")
        target_query = f"""
            SELECT DISTINCT entity_id,
                lower(country) || '|' || name_token AS blocking_key
            FROM training_targets,
            UNNEST(string_split({target_name}, ' ')) AS tokens(name_token)
            WHERE length(name_token) >= 3
              AND name_token NOT IN ({stop_words})
        """
        s1_query = f"""
            SELECT DISTINCT entity_id AS s1_id,
                lower(country) || '|' || name_token AS blocking_key
            FROM training_s1,
            UNNEST(string_split({s1_name}, ' ')) AS tokens(name_token)
            WHERE length(name_token) >= 3
              AND name_token NOT IN ({stop_words})
        """
    elif blocking_rule == "B2":
        target_name = normalized_text_sql("business_name")
        s1_name = normalized_text_sql("business_name")
        target_compact_name = f"replace({target_name}, ' ', '')"
        s1_compact_name = f"replace({s1_name}, ' ', '')"
        target_query = f"""
            SELECT DISTINCT entity_id,
                substr({target_compact_name}, 1, 4) AS blocking_key
            FROM training_targets
            WHERE length({target_compact_name}) >= 4
        """
        s1_query = f"""
            SELECT DISTINCT entity_id AS s1_id,
                substr({s1_compact_name}, 1, 4) AS blocking_key
            FROM training_s1
            WHERE length({s1_compact_name}) >= 4
        """
    elif blocking_rule == "B3":
        stop_words = sql_string_list(ADDRESS_STOP_WORDS)
        target_address = normalized_text_sql("business_address")
        s1_address = normalized_text_sql("business_address")
        target_query = f"""
            SELECT DISTINCT entity_id,
                lower(country) || '|' || address_token AS blocking_key
            FROM training_targets,
            UNNEST(string_split({target_address}, ' ')) AS tokens(address_token)
            WHERE length(address_token) >= 3
              AND address_token NOT IN ({stop_words})
        """
        s1_query = f"""
            SELECT DISTINCT entity_id AS s1_id,
                lower(country) || '|' || address_token AS blocking_key
            FROM training_s1,
            UNNEST(string_split({s1_address}, ' ')) AS tokens(address_token)
            WHERE length(address_token) >= 3
              AND address_token NOT IN ({stop_words})
        """
    else:
        raise ValueError(f"Unsupported blocking rule: {blocking_rule}")

    return target_query, s1_query


def generate_rule_candidates(
    connection: duckdb.DuckDBPyConnection,
    blocking_rule: str,
    maximum_records_per_key: int,
) -> int:
    """Generate and save candidates for one existing blocking rule."""
    for table_name in (
        "target_blocking_keys",
        "s1_blocking_keys",
        "allowed_blocking_keys",
        "candidate_pairs",
    ):
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")

    target_query, s1_query = rule_key_queries(blocking_rule)
    connection.execute(f"CREATE TEMP TABLE target_blocking_keys AS {target_query}")
    connection.execute(f"CREATE TEMP TABLE s1_blocking_keys AS {s1_query}")
    connection.execute(f"""
        CREATE TEMP TABLE allowed_blocking_keys AS
        SELECT blocking_key
        FROM target_blocking_keys
        GROUP BY blocking_key
        HAVING COUNT(DISTINCT entity_id) <= {int(maximum_records_per_key)}
    """)
    connection.execute("""
        CREATE TEMP TABLE candidate_pairs AS
        SELECT DISTINCT s1_keys.s1_id, target_keys.entity_id AS matched_id
        FROM s1_blocking_keys AS s1_keys
        JOIN allowed_blocking_keys USING (blocking_key)
        JOIN target_blocking_keys AS target_keys USING (blocking_key)
    """)

    candidate_table = f"rule_candidates_{blocking_rule.lower()}"
    connection.execute(f"DROP TABLE IF EXISTS {candidate_table}")
    connection.execute(f"""
        CREATE TEMP TABLE {candidate_table} AS
        SELECT
            candidates.s1_id,
            candidates.matched_id,
            sources.matched_source,
            '{blocking_rule}' AS blocking_rule
        FROM candidate_pairs AS candidates
        JOIN target_source_ids AS sources USING (matched_id)
    """)

    original_count = connection.execute(
        "SELECT COUNT(*) FROM candidate_pairs"
    ).fetchone()[0]
    saved_count = connection.execute(
        f"SELECT COUNT(*) FROM {candidate_table}"
    ).fetchone()[0]
    if saved_count != original_count:
        raise ValueError(
            f"{blocking_rule}: source mapping changed the candidate count "
            f"({original_count:,} to {saved_count:,})."
        )
    return saved_count


def validate_and_export(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Union rule outputs, validate them, then write the final CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    connection.execute("DROP TABLE IF EXISTS all_rule_candidate_pairs")
    connection.execute("""
        CREATE TEMP TABLE all_rule_candidate_pairs AS
        SELECT * FROM rule_candidates_b1
        UNION ALL
        SELECT * FROM rule_candidates_b2
        UNION ALL
        SELECT * FROM rule_candidates_b3
    """)
    connection.execute("DROP TABLE IF EXISTS final_candidate_pairs")
    connection.execute("""
        CREATE TEMP TABLE final_candidate_pairs AS
        SELECT
            s1_id,
            matched_id,
            matched_source,
            string_agg(DISTINCT blocking_rule, '|' ORDER BY blocking_rule)
                AS blocking_rule
        FROM all_rule_candidate_pairs
        GROUP BY s1_id, matched_id, matched_source
    """)

    rule_counts = {
        rule: connection.execute(
            f"SELECT COUNT(*) FROM rule_candidates_{rule.lower()}"
        ).fetchone()[0]
        for rule in ("B1", "B2", "B3")
    }
    combined_count = connection.execute(
        "SELECT COUNT(*) FROM final_candidate_pairs"
    ).fetchone()[0]
    duplicate_rule_pairs_removed = sum(rule_counts.values()) - combined_count
    s1_record_count = connection.execute(
        "SELECT COUNT(*) FROM training_s1"
    ).fetchone()[0]
    represented_s1_count = connection.execute(
        "SELECT COUNT(DISTINCT s1_id) FROM final_candidate_pairs"
    ).fetchone()[0]

    ground_truth_pair_count = connection.execute(
        "SELECT COUNT(*) FROM ground_truth"
    ).fetchone()[0]
    retained_true_pair_count = connection.execute("""
        SELECT COUNT(*)
        FROM ground_truth AS truth
        JOIN final_candidate_pairs AS candidates
          ON truth.s1_id = candidates.s1_id
         AND truth.matched_id = candidates.matched_id
    """).fetchone()[0]
    ground_truth_s1_count = connection.execute(
        "SELECT COUNT(DISTINCT s1_id) FROM ground_truth"
    ).fetchone()[0]
    retained_true_s1_count = connection.execute("""
        SELECT COUNT(DISTINCT truth.s1_id)
        FROM ground_truth AS truth
        WHERE EXISTS (
            SELECT 1
            FROM final_candidate_pairs AS candidates
            WHERE candidates.s1_id = truth.s1_id
              AND candidates.matched_id = truth.matched_id
        )
    """).fetchone()[0]

    invalid_s1_count = connection.execute("""
        SELECT COUNT(*)
        FROM final_candidate_pairs AS candidates
        LEFT JOIN training_s1 AS source1
          ON candidates.s1_id = source1.entity_id
        WHERE source1.entity_id IS NULL
    """).fetchone()[0]
    invalid_target_count = connection.execute("""
        SELECT COUNT(*)
        FROM final_candidate_pairs AS candidates
        LEFT JOIN target_source_ids AS sources
          ON candidates.matched_id = sources.matched_id
         AND candidates.matched_source = sources.matched_source
        WHERE sources.matched_id IS NULL
    """).fetchone()[0]
    invalid_source_count = connection.execute("""
        SELECT COUNT(*)
        FROM final_candidate_pairs
        WHERE matched_source NOT IN ('S2', 'S3')
           OR matched_source IS NULL
    """).fetchone()[0]
    null_identifier_count = connection.execute("""
        SELECT COUNT(*)
        FROM final_candidate_pairs
        WHERE s1_id IS NULL OR matched_id IS NULL OR blocking_rule IS NULL
    """).fetchone()[0]

    if any((invalid_s1_count, invalid_target_count, invalid_source_count,
            null_identifier_count)):
        raise ValueError(
            "Candidate validation failed: "
            f"invalid S1 IDs={invalid_s1_count:,}, "
            f"invalid target IDs={invalid_target_count:,}, "
            f"invalid source labels={invalid_source_count:,}, "
            f"null required values={null_identifier_count:,}."
        )

    candidate_counts_by_s1 = connection.execute("""
        SELECT COUNT(candidates.matched_id) AS candidate_count
        FROM training_s1 AS source1
        LEFT JOIN final_candidate_pairs AS candidates
          ON source1.entity_id = candidates.s1_id
        GROUP BY source1.entity_id
    """).fetchall()
    candidate_distribution = [row[0] for row in candidate_counts_by_s1]
    distribution = {
        "mean_candidates_per_s1": float(pd.Series(candidate_distribution).mean()),
        "median_candidates_per_s1": float(pd.Series(candidate_distribution).median()),
        "p95_candidates_per_s1": float(pd.Series(candidate_distribution).quantile(0.95)),
        "p99_candidates_per_s1": float(pd.Series(candidate_distribution).quantile(0.99)),
        "maximum_candidates_per_s1": int(max(candidate_distribution, default=0)),
    }

    temporary_output = output_path.with_name(output_path.name + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()
    connection.execute(f"""
        COPY (
            SELECT s1_id, matched_id, matched_source, blocking_rule
            FROM final_candidate_pairs
        )
        TO {sql_path(temporary_output)}
        (HEADER, DELIMITER ',')
    """)

    pandas_row_count = 0
    for chunk in pd.read_csv(temporary_output, dtype="string", chunksize=250_000):
        if list(chunk.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected CSV columns: {list(chunk.columns)}")
        if chunk[OUTPUT_COLUMNS].isna().any().any():
            raise ValueError("Export contains an empty required value.")
        if not chunk["matched_source"].isin(["S2", "S3"]).all():
            raise ValueError("Export contains an invalid matched_source value.")
        pandas_row_count += len(chunk)

    if pandas_row_count != combined_count:
        raise ValueError(
            f"DuckDB counted {combined_count:,} rows but pandas read "
            f"{pandas_row_count:,} rows."
        )

    if output_path.exists():
        output_path.unlink()
    os.replace(temporary_output, output_path)

    sample_rows = connection.execute(
        "SELECT s1_id, matched_id, matched_source, blocking_rule "
        "FROM final_candidate_pairs LIMIT 5"
    ).df().to_dict(orient="records")
    result = {
        "rule_candidate_counts": rule_counts,
        "combined_unique_candidate_count": combined_count,
        "duplicate_rule_pairs_removed": duplicate_rule_pairs_removed,
        "s1_records": s1_record_count,
        "s1_entities_represented": represented_s1_count,
        "average_candidates_per_s1": combined_count / s1_record_count,
        **distribution,
        "retained_true_match_pairs": retained_true_pair_count,
        "total_true_match_pairs": ground_truth_pair_count,
        "pair_level_candidate_recall": (
            retained_true_pair_count / ground_truth_pair_count
            if ground_truth_pair_count else None
        ),
        "s1_entities_with_true_matches": ground_truth_s1_count,
        "s1_entities_with_at_least_one_retained_true_match": retained_true_s1_count,
        "entity_level_candidate_recall": (
            retained_true_s1_count / ground_truth_s1_count
            if ground_truth_s1_count else None
        ),
        "csv_rows_read_by_pandas": pandas_row_count,
        "csv_file": str(output_path.resolve()),
        "csv_size_bytes": output_path.stat().st_size,
        "sample_rows": sample_rows,
    }
    return result


def run_blocking_experiment(
    data_dir: Path,
    output_path: Path,
    maximum_records_per_key: int = 100,
    memory_limit: str = "8GB",
    overwrite: bool = False,
    split: str = "train",
) -> dict[str, Any]:
    """Run B1, B2, and B3 against training or test sources and export."""
    if split not in {"train", "test"}:
        raise ValueError("Split must be either 'train' or 'test'.")
    if maximum_records_per_key < 1:
        raise ValueError("The maximum records per blocking key must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    data_dir = data_dir.resolve()
    output_path = output_path.resolve()
    temporary_dir = Path(tempfile.mkdtemp(prefix="blocking-work-"))
    source_files = SOURCE_FILES if split == "train" else TEST_SOURCE_FILES
    ground_truth_file = GROUND_TRUTH_FILE if split == "train" else None
    cleaned_paths = clean_input_files(
        data_dir, temporary_dir, source_files, ground_truth_file
    )
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute(
        f"SET temp_directory = {sql_path(temporary_dir / 'duckdb_spill')}"
    )

    try:
        create_source_views(connection, cleaned_paths, source_files)
        create_target_tables(connection)
        if ground_truth_file is not None:
            create_ground_truth(connection, cleaned_paths[ground_truth_file])
        else:
            connection.execute("""
                CREATE OR REPLACE TEMP TABLE ground_truth (
                    s1_id VARCHAR,
                    matched_id VARCHAR
                )
            """)

        country_rows = []
        for source in ("S1", "S2", "S3"):
            country_rows.extend(connection.execute(f"""
                SELECT '{source}' AS source,
                       coalesce(country, 'Missing country') AS country,
                       count(*) AS records
                FROM training_{source.lower()}
                GROUP BY country
            """).fetchall())

        per_rule_counts = {}
        for blocking_rule in ("B1", "B2", "B3"):
            per_rule_counts[blocking_rule] = generate_rule_candidates(
                connection,
                blocking_rule,
                maximum_records_per_key,
            )

        result = validate_and_export(connection, output_path, overwrite)
        result["source_record_counts"] = {
            source: connection.execute(
                f"SELECT COUNT(*) FROM training_{source.lower()}"
            ).fetchone()[0]
            for source in ("S1", "S2", "S3")
        }
        result["business_records_by_country_and_source"] = [
            {"source": row[0], "country": row[1], "records": row[2]}
            for row in country_rows
        ]
        result["maximum_records_per_blocking_key"] = maximum_records_per_key
        result["data_split"] = split
        result["recall_available"] = ground_truth_file is not None
        result["rule_candidate_counts"] = per_rule_counts
        return result
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate candidate business pairs from B1, B2, and B3."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing the source files for the selected split.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Use train_* files with ground truth, or test_* files without labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/candidate_pairs.csv"),
        help="Candidate CSV output path; default: experiments/candidate_pairs.csv.",
    )
    parser.add_argument(
        "--maximum-records-per-key",
        type=int,
        default=100,
        help="Skip a blocking key when it matches more target records than this.",
    )
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit, for example 8GB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing candidate output file after successful validation.",
    )
    arguments = parser.parse_args()

    result = run_blocking_experiment(
        data_dir=arguments.data_dir,
        output_path=arguments.output,
        maximum_records_per_key=arguments.maximum_records_per_key,
        memory_limit=arguments.memory_limit,
        overwrite=arguments.overwrite,
        split=arguments.split,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
