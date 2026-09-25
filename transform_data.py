from pathlib import Path
from typing import Tuple, List
import os
import pandas as pd

# Try to load settings from config if available, otherwise fallback
try:
    from src.config import DATA_ROOT
    BASE_DIR = DATA_ROOT
except Exception:
    BASE_DIR = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations")

MONTHLY_PATH  = BASE_DIR / "monthly_app_summary.xlsx"
SETTINGS_PATH = BASE_DIR / "Settings.xlsx"
FINAL_PATH    = BASE_DIR / "monthly_final_summary.xlsx"
# ─────────────────────────────────────────────────────────────────────────────

def generate_final_summary() -> Tuple[List[int], List[int]]:
    # 2. Load raw monthly data
    df = pd.read_excel(MONTHLY_PATH, dtype={'ReportMonth': str})
    df.rename(columns={'ReportMonth': 'Month'}, inplace=True)

    # 3. Load Country → Country Group
    df_country = pd.read_excel(
        SETTINGS_PATH, sheet_name="Country", dtype={'CountryId': int}
    )

    # 4. Load TagGroup → Region
    df_region = pd.read_excel(SETTINGS_PATH, sheet_name="Region")

    # 5. Merge in Country Grouped; mark which were unmapped
    df = df.merge(
        df_country[['CountryId','Country Grouped']],
        on='CountryId', how='left'
    )
    df['country_was_unmapped'] = df['Country Grouped'].isna()

    # 6. Pivot Region sheet to long form and merge; mark which were unmapped
    region_melt = (
        df_region
        .melt(id_vars=['id'], var_name='Country Grouped', value_name='RegionName')
        .rename(columns={'id':'TagGroupID'})
    )
    df = df.merge(
        region_melt[['TagGroupID','Country Grouped','RegionName']],
        on=['TagGroupID','Country Grouped'], how='left'
    )
    df['region_was_unmapped'] = df['RegionName'].isna()

    # Capture missing IDs before filling
    missing_countries = sorted(df.loc[df['country_was_unmapped'], 'CountryId'].dropna().unique().astype(int).tolist())
    missing_tags      = sorted(df.loc[df['region_was_unmapped'], 'TagGroupID'].dropna().unique().astype(int).tolist())

    # 7. Fill unmapped with “Other”
    df['Country Grouped'] = df['Country Grouped'].fillna('Other')
    df['RegionName']      = df['RegionName'].fillna('Other')

    # 8. Build unmapped‐CountryIDs per month
    unmapped_countries = (
        df[df['country_was_unmapped']]
        .groupby('Month')['CountryId']
        .unique()
        .reset_index()
    )
    unmapped_countries['UnmappedCountryIDs'] = (
        unmapped_countries['CountryId']
        .apply(lambda arr: ",".join(map(str, sorted(arr))))
    )
    unmapped_countries['Country Grouped'] = 'Other'
    unmapped_countries.drop(columns=['CountryId'], inplace=True)

    # 9. Build unmapped‐TagGroupIDs per month+country_group
    unmapped_tags = (
        df[df['region_was_unmapped']]
        .groupby(['Month','Country Grouped'])['TagGroupID']
        .unique()
        .reset_index()
    )
    unmapped_tags['UnmappedTagGroupIDs'] = (
        unmapped_tags['TagGroupID']
        .apply(lambda arr: ",".join(map(str, sorted(arr))))
    )
    unmapped_tags.drop(columns=['TagGroupID'], inplace=True)

    # 10. Aggregate (sum) across TagGroups per Month/Country/Region
    agg = df.groupby(
        ['Month','Country Grouped','RegionName'], as_index=False
    )[
        ['ReceivedApps','ReceivedStudents','SubmittedApps','SubmittedStudents']
    ].sum()

    # 11. Merge in both unmapped lists
    agg = (
        agg
        .merge(unmapped_countries, on=['Month','Country Grouped'], how='left')
        .merge(unmapped_tags,      on=['Month','Country Grouped'], how='left')
        .fillna({'UnmappedCountryIDs':'','UnmappedTagGroupIDs':''})
    )

    # 12. Final rename & write
    agg.rename(columns={'RegionName':'Name of Region'}, inplace=True)
    agg.to_excel(FINAL_PATH, index=False)
    
    if missing_countries or missing_tags:
        print(f"⚠️  Warning: Found unmapped items (grouped as 'Other').")
        if missing_countries: print(f"   Missing CountryIDs: {missing_countries}")
        if missing_tags:      print(f"   Missing TagGroupIDs: {missing_tags}")
    else:
        print(f"✅ Final summary written to {FINAL_PATH} (All mapped)")
    
    return missing_countries, missing_tags

if __name__ == "__main__":
    generate_final_summary()