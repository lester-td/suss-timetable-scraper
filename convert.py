from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import camelot
import pandas as pd
import pdfplumber


CORE_REQUIRED_COLUMNS = [
    "SCHOOL / CENTRE",
    "COURSE CODE",
    "CRN / TG",
    "SEMESTER TYPE",
    "DELIVERY / EXAM MODE",
    "DATE",
    "START",
    "END",
]

PREFERRED_ORDER = [
    "POSTGRADUATE",
    "SCHOOL / CENTRE",
    "COURSE CODE",
    "CRN / TG",
    "SEMESTER TYPE",
    "DELIVERY / EXAM MODE",
    "DAY",
    "DATE",
    "START",
    "END",
    "AVAILABLE AS GSP100/UNE500",
    "REMARKS",
]

TEXT_VERIFY_COLUMNS = [
    "COURSE CODE",
    "CRN / TG",
    "DELIVERY / EXAM MODE",
    "DATE",
    "START",
    "END",
]


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_for_match(value: object) -> str:
    text = clean_text(value).upper()
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_compact_match(value: object) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


def normalize_columns(columns: list[str]) -> list[str]:
    cleaned = [clean_text(col) for col in columns]

    rename_map: dict[str, str] = {}

    return [rename_map.get(col, col) for col in cleaned]


def is_repeated_header_row(row: pd.Series, expected_columns: list[str]) -> bool:
    row_values = [clean_text(x) for x in row.tolist()]
    return row_values == expected_columns


def list_pdf_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )


def choose_pdf_file() -> Path:
    current_dir = Path.cwd()
    pdf_files = list_pdf_files(current_dir)

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the current directory.")

    print("PDF files found in this directory:\n")
    for i, pdf_file in enumerate(pdf_files, start=1):
        print(f"{i}. {pdf_file.name}")

    while True:
        choice = input("\nEnter the number of the PDF to process: ").strip()

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(choice)
        if 1 <= index <= len(pdf_files):
            selected = pdf_files[index - 1]
            print(f"\nSelected PDF: {selected.name}")
            return selected

        print("Choice out of range. Try again.")


def next_available_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    counter = 2
    while True:
        candidate = base_path.with_name(f"{base_path.stem}-{counter}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_output_paths(pdf_path: Path) -> tuple[Path, Path, Path]:
    base_csv = pdf_path.with_suffix(".csv")
    output_csv = next_available_path(base_csv)

    run_stem = output_csv.stem
    summary_txt = output_csv.with_name(f"{run_stem}.verification_summary.txt")
    failures_csv = output_csv.with_name(f"{run_stem}.verification_failures.csv")

    return output_csv, summary_txt, failures_csv


def get_pdf_page_count(pdf_path: Path) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def merge_cell_values(left: object, right: object) -> str:
    a = clean_text(left)
    b = clean_text(right)

    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a

    return f"{a} {b}"


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    seen_blank = 0

    for col_idx in range(df.shape[1]):
        raw_name = df.columns[col_idx]
        col_name = clean_text(raw_name)

        if col_name == "":
            seen_blank += 1
            col_name = f"__blank__{seen_blank}"

        series = df.iloc[:, col_idx].map(clean_text)

        if col_name not in result.columns:
            result[col_name] = series
        else:
            result[col_name] = [
                merge_cell_values(existing, new)
                for existing, new in zip(result[col_name], series)
            ]

    blank_cols_to_drop: list[str] = []
    for col in result.columns:
        if col.startswith("__blank__") and result[col].map(clean_text).eq("").all():
            blank_cols_to_drop.append(col)

    if blank_cols_to_drop:
        result = result.drop(columns=blank_cols_to_drop)

    return result


def align_to_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CORE_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    metadata_cols = [col for col in df.columns if col.startswith("__")]
    normal_cols = [col for col in df.columns if col not in metadata_cols]

    ordered = [col for col in PREFERRED_ORDER if col in normal_cols]
    remaining = [col for col in normal_cols if col not in ordered]

    return df[ordered + remaining + metadata_cols]


def row_looks_like_header(row: list[str]) -> bool:
    normalized = [clean_text(x).upper() for x in row]

    header_hits = 0
    header_keywords = {
        "POSTGRADUATE",
        "SCHOOL / CENTRE",
        "COURSE CODE",
        "CRN / TG",
        "SEMESTER TYPE",
        "DELIVERY / EXAM MODE",
        "DAY",
        "DATE",
        "START",
        "END",
        "AVAILABLE AS GSP100/UNE500",
        "REMARKS",
    }

    for cell in normalized:
        if cell in header_keywords:
            header_hits += 1

    return header_hits >= 3


def extract_tables_from_pdf(pdf_path: Path) -> pd.DataFrame:
    total_pages = get_pdf_page_count(pdf_path)
    print(f"\n[1/2] Extracting tables from PDF: {pdf_path.name}")
    print(f"Detected {total_pages} page(s).")

    all_frames: list[pd.DataFrame] = []
    total_rows_kept = 0
    canonical_header: list[str] | None = None

    for page_num in range(1, total_pages + 1):
        print(f"\n--- Page {page_num}/{total_pages} ---")
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num),
            flavor="lattice",
            suppress_stdout=True,
        )
        print(f"Tables found by Camelot: {len(tables)}")

        if len(tables) == 0:
            continue

        for table_idx, table in enumerate(tables, start=1):
            report = getattr(table, "parsing_report", {}) or {}
            accuracy = report.get("accuracy", "n/a")
            whitespace = report.get("whitespace", "n/a")
            print(
                f"  Table {table_idx}: shape={table.shape}, "
                f"accuracy={accuracy}, whitespace={whitespace}"
            )

            raw_df = table.df.copy()

            if raw_df.empty:
                print("    Skipping: empty table.")
                continue

            cleaned_rows: list[list[str]] = []
            for _, row in raw_df.iterrows():
                cleaned = [clean_text(x) for x in row.tolist()]
                if any(cell for cell in cleaned):
                    cleaned_rows.append(cleaned)

            if not cleaned_rows:
                print("    Skipping: no usable rows.")
                continue

            first_row = cleaned_rows[0]

            if row_looks_like_header(first_row):
                header = normalize_columns(first_row)
                canonical_header = header
                data_rows = cleaned_rows[1:]
                print("    Header detected on this page.")
            else:
                if canonical_header is None:
                    print("    Skipping: no header detected and no prior header available.")
                    continue
                header = canonical_header
                data_rows = cleaned_rows
                print("    No header detected. Reusing previous header.")

            if not data_rows:
                print("    Skipping: no data rows after header handling.")
                continue

            header_width = len(header)
            normalized_rows: list[list[str]] = []
            for row in data_rows:
                if len(row) < header_width:
                    row = row + [""] * (header_width - len(row))
                elif len(row) > header_width:
                    row = row[:header_width]
                normalized_rows.append(row)

            body = pd.DataFrame(normalized_rows, columns=header)
            body = body.apply(lambda col: col.map(clean_text))

            rows_before_cleanup = len(body)

            body = body.loc[
                ~body.apply(lambda row: is_repeated_header_row(row, header), axis=1)
            ].copy()

            body = body.loc[
                ~body.apply(lambda row: all(clean_text(x) == "" for x in row), axis=1)
            ].copy()

            duplicate_headers = pd.Index(body.columns)[
                pd.Index(body.columns).duplicated()
            ].tolist()
            if duplicate_headers:
                print(f"    Duplicate headers detected: {duplicate_headers}")

            body = collapse_duplicate_columns(body)
            body = align_to_expected_columns(body)

            rows_after_cleanup = len(body)

            if body.empty:
                print("    Skipping: no usable rows after cleanup.")
                continue

            body["__source_page"] = str(page_num)
            body["__source_table"] = str(table_idx)

            all_frames.append(body)
            total_rows_kept += rows_after_cleanup

            print(
                f"    Rows kept: {rows_after_cleanup} "
                f"(before cleanup: {rows_before_cleanup})"
            )

    if not all_frames:
        raise RuntimeError("No tables were extracted from the PDF.")

    print("\nChecking columns before final concat...")
    for i, frame in enumerate(all_frames, start=1):
        if not frame.columns.is_unique:
            dupes = frame.columns[frame.columns.duplicated()].tolist()
            raise RuntimeError(f"Frame {i} still has duplicate columns: {dupes}")

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.columns = [clean_text(col) for col in final_df.columns]

    print(f"\nExtraction complete. Total extracted rows: {len(final_df)}")
    print(f"Total rows kept across all pages: {total_rows_kept}")

    return final_df


def run_basic_checks(df: pd.DataFrame) -> list[dict[str, str]]:
    print("\nRunning basic integrity checks...")
    failures: list[dict[str, str]] = []

    missing_columns = [col for col in CORE_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        failures.append(
            {
                "issue_type": "missing_columns",
                "details": ", ".join(missing_columns),
            }
        )
        print(f"Missing required core columns: {missing_columns}")
        return failures

    print("Required core columns present.")

    critical_cols = [
        col
        for col in ["COURSE CODE", "CRN / TG", "DATE", "START", "END"]
        if col in df.columns
    ]

    for col in critical_cols:
        blank_mask = df[col].map(clean_text).eq("")
        blank_count = int(blank_mask.sum())
        print(f"Blank values in {col}: {blank_count}")
        if blank_count:
            failures.append(
                {
                    "issue_type": "blank_critical_values",
                    "details": f"{col}: {blank_count} blank row(s)",
                }
            )

    visible_columns = [col for col in df.columns if not col.startswith("__")]
    duplicate_mask = df.duplicated(subset=visible_columns, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    print(f"Exact duplicate rows: {duplicate_count}")
    if duplicate_count:
        failures.append(
            {
                "issue_type": "duplicate_rows",
                "details": f"{duplicate_count} duplicate row(s) detected",
            }
        )

    if "DAY" in df.columns:
        invalid_date_count = 0
        day_date_mismatch_count = 0

        for row_idx, row in df.iterrows():
            date_text = clean_text(row.get("DATE", ""))
            day_text = normalize_for_match(row.get("DAY", ""))

            if not date_text or not day_text:
                continue

            try:
                expected_day = datetime.strptime(date_text, "%d/%m/%Y").strftime("%A").upper()
            except ValueError as exc:
                invalid_date_count += 1
                failures.append(
                    {
                        "issue_type": "invalid_date_format",
                        "details": f"row={row_idx + 2}, DATE={date_text}, error={exc}",
                    }
                )
                continue

            if day_text != expected_day:
                day_date_mismatch_count += 1
                failures.append(
                    {
                        "issue_type": "day_date_mismatch",
                        "details": (
                            f"row={row_idx + 2}, COURSE CODE={row.get('COURSE CODE', '')}, "
                            f"CRN / TG={row.get('CRN / TG', '')}, DATE={date_text}, "
                            f"DAY={row.get('DAY', '')}, EXPECTED={expected_day}"
                        ),
                    }
                )

        print(f"Invalid DATE rows: {invalid_date_count}")
        print(f"DAY vs DATE mismatches: {day_date_mismatch_count}")

    return failures


def verify_against_pdf_text(df: pd.DataFrame, pdf_path: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    print("\n[2/2] Verifying extracted rows against source PDF text...")
    print("Using stable fields only: COURSE CODE, CRN / TG, DELIVERY / EXAM MODE, DATE, START, END")
    print("Skipping SCHOOL / CENTRE, DAY, and REMARKS because PDF text wrapping/layout causes false mismatches.")

    page_groups = df.groupby("__source_page", sort=True)

    with pdfplumber.open(pdf_path) as pdf:
        for source_page, page_df in page_groups:
            page_number = int(source_page)
            page_text = pdf.pages[page_number - 1].extract_text() or ""
            normalized_page_text = normalize_for_compact_match(page_text)

            print(f"\n--- Verifying page {page_number} ---")
            print(f"Rows to verify on this page: {len(page_df)}")

            page_failures_before = len(failures)

            for row_idx, row in page_df.iterrows():
                missing_fields: list[str] = []

                for col in TEXT_VERIFY_COLUMNS:
                    if col not in row.index:
                        continue

                    value = clean_text(row[col])
                    if not value:
                        continue

                    token = normalize_for_compact_match(value)
                    if token and token not in normalized_page_text:
                        missing_fields.append(col)

                if missing_fields:
                    failures.append(
                        {
                            "issue_type": "pdf_text_mismatch",
                            "details": (
                                f"row={row_idx + 2}, page={page_number}, "
                                f"COURSE CODE={row.get('COURSE CODE', '')}, "
                                f"CRN / TG={row.get('CRN / TG', '')}, "
                                f"DATE={row.get('DATE', '')}, "
                                f"missing_fields={missing_fields}"
                            ),
                        }
                    )

            page_failure_count = len(failures) - page_failures_before
            page_success_count = len(page_df) - page_failure_count
            print(f"Matched on page {page_number}: {page_success_count}/{len(page_df)} row(s)")

    return failures


def write_outputs(
    df: pd.DataFrame,
    verification_failures: list[dict[str, str]],
    output_csv: Path,
    failures_csv: Path,
    summary_txt: Path,
    pdf_path: Path,
) -> None:
    output_df = df.drop(columns=["__source_page", "__source_table"], errors="ignore").copy()
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"PDF: {pdf_path.name}",
        f"CSV: {output_csv.name}",
        f"Total extracted rows: {len(output_df)}",
        f"Verification issues found: {len(verification_failures)}",
    ]

    if verification_failures:
        failures_df = pd.DataFrame(verification_failures)
        failures_df.to_csv(failures_csv, index=False, encoding="utf-8-sig")
        summary_lines.append(f"Failure report: {failures_csv.name}")
        summary_lines.append("Status: VERIFICATION FAILED")
    else:
        summary_lines.append("Failure report: none")
        summary_lines.append(
            "Status: VERIFIED (automated) - all configured checks passed and every extracted row matched back to source PDF text."
        )

    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nSaved cleaned CSV to: {output_csv.name}")
    print(f"Saved verification summary to: {summary_txt.name}")

    if verification_failures:
        print(f"Saved verification failures to: {failures_csv.name}")


def main() -> None:
    pdf_path = choose_pdf_file()
    output_csv, summary_txt, failures_csv = build_output_paths(pdf_path)

    extracted_df = extract_tables_from_pdf(pdf_path)

    basic_failures = run_basic_checks(extracted_df)
    text_failures = verify_against_pdf_text(extracted_df, pdf_path)
    verification_failures = basic_failures + text_failures

    write_outputs(
        df=extracted_df,
        verification_failures=verification_failures,
        output_csv=output_csv,
        failures_csv=failures_csv,
        summary_txt=summary_txt,
        pdf_path=pdf_path,
    )

    print("\n==================== FINAL STATUS ====================")
    if verification_failures:
        print("VERIFICATION FAILED.")
        print("The CSV was still written, but one or more checks did not pass.")
        print(f"Review: {failures_csv.name}")
        print(f"Summary: {summary_txt.name}")
    else:
        print("VERIFIED (automated).")
        print("All configured checks passed.")
        print("Every extracted row matched back to the source PDF text.")
        print(f"CSV ready: {output_csv.name}")
        print(f"Summary: {summary_txt.name}")


if __name__ == "__main__":
    main()