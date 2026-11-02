"""
Data ingestion and validation module.

This module reads raw CSV data and validates it using Great Expectations
before passing it to the feature engineering pipeline.

Usage:
    python -m src.ingest --raw data/raw/train.csv
    python -m src.ingest --raw data/raw/test.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import great_expectations as gx
from great_expectations.expectations.expectation import ExpectationConfiguration


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

EXPECTED_COLUMNS = [
    'member_id', 'age', 'gender', 'race', 'state', 'is_rural', 'plan_type',
    'dual_eligible', 'mental_health_diagnosis', 'tenure_years', 'chronic_conditions',
    'claims_last_12mo', 'denied_claims_last_12mo', 'premium_increase_pct',
    'service_calls_last_12mo', 'last_satisfaction_score', 'renewal_decision_date',
    'days_to_renewal_decision', 'cancellation_form_started', 'churned'
]

# Columns that are allowed to have null values
NULLABLE_COLUMNS = ['last_satisfaction_score']

# Valid categorical values
VALID_GENDERS = ['M', 'F', 'X']
VALID_STATES = ['OH', 'IN', 'KY', 'WV']
VALID_PLAN_TYPES = ['commercial_group', 'individual_marketplace', 'medicare_advantage']
VALID_RACES = ['white', 'black', 'hispanic', 'asian', 'other']


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def create_expectation_suite():
    """Create a Great Expectations suite with all validation rules."""
    
    context = gx.get_context()
    suite_name = "churn_data_validation"
    
    # Create or get the expectation suite
    try:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    except Exception:
        # Suite might already exist
        suite = context.suites.get(suite_name)
    
    expectations = []
    
    # -------------------------------------------------------------------------
    # 1. SCHEMA VALIDATION - All expected columns exist
    # -------------------------------------------------------------------------
    for col in EXPECTED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnToExist(column=col)
        )
    
    # -------------------------------------------------------------------------
    # 2. PRIMARY KEY - member_id validation
    # -------------------------------------------------------------------------
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeUnique(column='member_id')
    )
    expectations.append(
        gx.expectations.ExpectColumnValuesToNotBeNull(column='member_id')
    )
    expectations.append(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column='member_id',
            regex=r'^M\d{7}$'  # M followed by 7 digits
        )
    )
    
    # -------------------------------------------------------------------------
    # 3. TARGET VARIABLE - churned validation
    # -------------------------------------------------------------------------
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='churned',
            value_set=[0, 1]
        )
    )
    expectations.append(
        gx.expectations.ExpectColumnValuesToNotBeNull(column='churned')
    )
    
    # -------------------------------------------------------------------------
    # 4. NUMERIC RANGES
    # -------------------------------------------------------------------------
    # Age: 18-100
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='age',
            min_value=18,
            max_value=100
        )
    )
    
    # Tenure: >= 0
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='tenure_years',
            min_value=0,
            max_value=100
        )
    )
    
    # Claims: >= 0
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='claims_last_12mo',
            min_value=0,
            max_value=1000
        )
    )
    
    # Denied claims: >= 0
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='denied_claims_last_12mo',
            min_value=0,
            max_value=1000
        )
    )
    
    # Premium increase: reasonable range
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='premium_increase_pct',
            min_value=-50,
            max_value=100
        )
    )
    
    # Service calls: >= 0
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='service_calls_last_12mo',
            min_value=0,
            max_value=500
        )
    )
    
    # Satisfaction score: 1-5 (when not null)
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column='last_satisfaction_score',
            min_value=1,
            max_value=5
        )
    )
    
    # -------------------------------------------------------------------------
    # 5. CATEGORICAL VALUES
    # -------------------------------------------------------------------------
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='gender',
            value_set=VALID_GENDERS
        )
    )
    
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='state',
            value_set=VALID_STATES
        )
    )
    
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='plan_type',
            value_set=VALID_PLAN_TYPES
        )
    )
    
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='race',
            value_set=VALID_RACES
        )
    )
    
    # -------------------------------------------------------------------------
    # 6. BOOLEAN FIELDS
    # -------------------------------------------------------------------------
    for bool_col in ['is_rural', 'dual_eligible', 'mental_health_diagnosis', 'chronic_conditions']:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=bool_col,
                value_set=[True, False]
            )
        )
    
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column='cancellation_form_started',
            value_set=[0, 1]
        )
    )
    
    # -------------------------------------------------------------------------
    # 7. NON-NULL CONSTRAINTS (for columns that shouldn't be null)
    # -------------------------------------------------------------------------
    non_null_columns = [col for col in EXPECTED_COLUMNS if col not in NULLABLE_COLUMNS]
    for col in non_null_columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )
    
    # -------------------------------------------------------------------------
    # 8. ROW COUNT - Sanity check
    # -------------------------------------------------------------------------
    expectations.append(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=100,
            max_value=1000000
        )
    )
    
    # Add all expectations to suite
    for exp in expectations:
        suite.add_expectation(exp)
    
    return context, suite


def validate_dataframe(df: pd.DataFrame, context, suite) -> dict:
    """
    Validate a DataFrame against the expectation suite.
    
    Returns:
        dict with 'success' (bool) and 'results' (validation details)
    """
    # Create a batch from the dataframe
    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="churn_data")
    
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # Run validation
    validation_result = batch.validate(suite)
    
    return {
        'success': validation_result.success,
        'results': validation_result.results,
        'statistics': validation_result.statistics
    }


def validate_data(filepath: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Load and validate a CSV file.
    
    Args:
        filepath: Path to the CSV file
        verbose: Whether to print validation results
        
    Returns:
        Tuple of (DataFrame, validation_results)
        
    Raises:
        ValueError: If validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if verbose:
        print(f"Loading data from {filepath}...")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"\nRunning validation checks...")
    
    # Create expectation suite and validate
    context, suite = create_expectation_suite()
    results = validate_dataframe(df, context, suite)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Success: {'✓ PASSED' if results['success'] else '✗ FAILED'}")
        print(f"Expectations Evaluated: {results['statistics']['evaluated_expectations']}")
        print(f"Successful: {results['statistics']['successful_expectations']}")
        print(f"Failed: {results['statistics']['unsuccessful_expectations']}")
        
        if not results['success']:
            print(f"\n{'='*60}")
            print("FAILED EXPECTATIONS:")
            print(f"{'='*60}")
            for result in results['results']:
                if not result.success:
                    exp_type = result.expectation_config.type
                    kwargs = result.expectation_config.kwargs
                    print(f"\n✗ {exp_type}")
                    print(f"  Config: {kwargs}")
                    if hasattr(result, 'result') and result.result:
                        print(f"  Result: {result.result}")
    
    if not results['success']:
        raise ValueError(
            f"Data validation failed with {results['statistics']['unsuccessful_expectations']} "
            f"failed expectations. See details above."
        )
    
    if verbose:
        print(f"\n✓ All validation checks passed!")
    
    return df, results


# =============================================================================
# SIMPLE VALIDATION (fallback without Great Expectations complexity)
# =============================================================================

def validate_data_simple(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Simple validation without Great Expectations (more reliable).
    
    Args:
        filepath: Path to the CSV file
        verbose: Whether to print validation results
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if verbose:
        print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"\nRunning validation checks...")
    
    errors = []
    warnings = []
    
    # 1. Check all expected columns exist
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # 2. Check for unexpected columns
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        warnings.append(f"Unexpected columns (will be ignored): {extra_cols}")
    
    # 3. Check member_id uniqueness
    if df['member_id'].duplicated().any():
        n_dups = df['member_id'].duplicated().sum()
        errors.append(f"member_id has {n_dups} duplicate values")
    
    # 4. Check member_id format
    invalid_ids = ~df['member_id'].str.match(r'^M\d{7}$')
    if invalid_ids.any():
        errors.append(f"member_id has {invalid_ids.sum()} values not matching format M#######")
    
    # 5. Check churned values
    invalid_churned = ~df['churned'].isin([0, 1])
    if invalid_churned.any():
        errors.append(f"churned has {invalid_churned.sum()} values not in [0, 1]")
    
    # 6. Check numeric ranges
    if (df['age'] < 18).any() or (df['age'] > 100).any():
        errors.append(f"age has values outside range [18, 100]")
    
    if (df['tenure_years'] < 0).any():
        errors.append(f"tenure_years has negative values")
    
    if (df['claims_last_12mo'] < 0).any():
        errors.append(f"claims_last_12mo has negative values")
    
    if (df['denied_claims_last_12mo'] < 0).any():
        errors.append(f"denied_claims_last_12mo has negative values")
    
    if (df['service_calls_last_12mo'] < 0).any():
        errors.append(f"service_calls_last_12mo has negative values")
    
    # Check satisfaction score range (only for non-null values)
    valid_scores = df['last_satisfaction_score'].dropna()
    if ((valid_scores < 1) | (valid_scores > 5)).any():
        errors.append(f"last_satisfaction_score has values outside range [1, 5]")
    
    # 7. Check categorical values
    invalid_gender = ~df['gender'].isin(VALID_GENDERS)
    if invalid_gender.any():
        bad_vals = df.loc[invalid_gender, 'gender'].unique()
        errors.append(f"gender has invalid values: {bad_vals}")
    
    invalid_state = ~df['state'].isin(VALID_STATES)
    if invalid_state.any():
        bad_vals = df.loc[invalid_state, 'state'].unique()
        errors.append(f"state has invalid values: {bad_vals}")
    
    invalid_plan = ~df['plan_type'].isin(VALID_PLAN_TYPES)
    if invalid_plan.any():
        bad_vals = df.loc[invalid_plan, 'plan_type'].unique()
        errors.append(f"plan_type has invalid values: {bad_vals}")
    
    invalid_race = ~df['race'].isin(VALID_RACES)
    if invalid_race.any():
        bad_vals = df.loc[invalid_race, 'race'].unique()
        errors.append(f"race has invalid values: {bad_vals}")
    
    # 8. Check non-null constraints
    for col in EXPECTED_COLUMNS:
        if col not in NULLABLE_COLUMNS:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"{col} has {null_count} null values")
    
    # 9. Check row count
    if len(df) < 100:
        errors.append(f"Too few rows: {len(df)} (expected >= 100)")
    
    # 10. Business logic: denied_claims <= claims
    invalid_denials = df['denied_claims_last_12mo'] > df['claims_last_12mo']
    if invalid_denials.any():
        errors.append(f"{invalid_denials.sum()} rows have denied_claims > total claims")
    
    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        
        if warnings:
            print("\nWARNINGS:")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        if errors:
            print("\nERRORS:")
            for e in errors:
                print(f"  ✗ {e}")
            print(f"\n✗ Validation FAILED with {len(errors)} errors")
        else:
            print(f"\n✓ All validation checks passed!")
    
    # Always raise if there are errors
    if errors:
        raise ValueError(f"Data validation failed with {len(errors)} errors")
    
    return df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ingest and validate raw data for churn prediction pipeline'
    )
    parser.add_argument(
        '--raw',
        type=str,
        required=True,
        help='Path to raw CSV file (e.g., data/raw/train.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save validated data (optional)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple validation instead of Great Expectations'
    )
    
    args = parser.parse_args()
    
    try:
        if args.simple:
            df = validate_data_simple(args.raw)
        else:
            # Try Great Expectations first, fall back to simple
            try:
                df, _ = validate_data(args.raw)
            except Exception as e:
                print(f"\nGreat Expectations validation encountered an issue: {e}")
                print("Falling back to simple validation...\n")
                df = validate_data_simple(args.raw)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nValidated data saved to {output_path}")
        
        print(f"\n✓ Ingestion complete: {len(df):,} rows validated")
        return 0
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
