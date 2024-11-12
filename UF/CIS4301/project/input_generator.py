import pandas as pd
from os.path import exists
from os import makedirs


sites = set()
chemicals = {}
next_chemical_ID = 1
release_ID = 0
years = range(1987, 2024)


def process_row_release(row: pd.core.series.Series, chemID: str) -> str:
    global release_ID
    release_ID += 1
    facility = row["Facility"].replace("'", "''")
    industry = row["Industry"].replace("'", "''")
    return f"INSERT INTO Releases VALUES ({release_ID}, {row['Year']}, '{facility}', '{industry}', {chemID}, '{row['TRIFID']}', '{row['Unit']}', {row['Amount']});\n"


def process_row_site(row: pd.core.series.Series) -> tuple[str, str]:
    # Check if already in data
    if row["TRIFID"] in sites:
        return None

    # Add to data
    sites.add(row["TRIFID"])
    street = row["Street"].replace("'", "''")
    city = row["City"].replace("'", "''")
    county = row["County"].replace("'", "''")
    return f"INSERT INTO Sites VALUES ('{row['TRIFID']}', {row['FRSID'] if not pd.isna(row['FRSID']) else 'Null'}, {row['Longitude']}, {row['Latitude']}, '{street}', '{city}', '{row['State']}', '{county}', '{row['Zip']}');\n"


def process_row_chemical(row: pd.core.series.Series) -> tuple[str, str]:
    global next_chemical_ID
    # Check if already in data
    if row["Chemical"] in chemicals.keys():
        return chemicals[row["Chemical"]], None

    # Add to data
    chemicals[row["Chemical"]] = next_chemical_ID
    next_chemical_ID += 1
    chemical = row["Chemical"].replace("'", "''")
    return (
        chemicals[row["Chemical"]],
        f"INSERT INTO Chemicals VALUES ({chemicals[row['Chemical']]}, '{chemical}', NULL);\n",
    )


def process_row(row: pd.core.series.Series, out_file) -> list[str]:
    chemID, chem_cmd = process_row_chemical(row)
    site_cmd = process_row_site(row)
    release_cmd = process_row_release(row, chemID)

    # Write to file
    if chem_cmd:
        out_file.write(chem_cmd)
    if site_cmd:
        out_file.write(site_cmd)
    out_file.write(release_cmd)


def main():
    folder_path = "insert_cmds/TRI/"
    if not exists(folder_path):
        makedirs(folder_path)

    for year in years:
        df = pd.read_csv(
            f"data/{year}_us.csv",
            skiprows=1,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 22, 36, 49, 64],
            names=[
                "Year",
                "TRIFID",
                "FRSID",
                "Facility",
                "Street",
                "City",
                "County",
                "State",
                "Zip",
                "Latitude",
                "Longitude",
                "Industry",
                "Chemical",
                "Unit",
                "Amount",
            ],
        )

        # Parse rows
        with open(
            f"{folder_path}insert_cmds_{year}.sql", "w", encoding="utf-8"
        ) as out_file:
            # Ignore '&'s
            out_file.write("SET DEFINE OFF\n")
            for _, row in df.iterrows():
                process_row(row, out_file)


if __name__ == "__main__":
    main()
