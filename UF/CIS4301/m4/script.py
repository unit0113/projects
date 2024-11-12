from operator import itemgetter

USECOLS = [0, 1, 2, 3, 4, 5, 6, 8, 11, 13, -6, -5, -4, -3, -2, -1]

with open("UF\CIS4301\m4\ToyCarOrdersAndSales Insert Commands.sql", "w") as write_file:
    # Ignore '&'
    write_file.write("SET DEFINE OFF\n")
    with open("UF\CIS4301\m4\Auto Sales data.csv", "r") as read_file:
        for line in read_file.readlines()[1:]:
            # Split data, remove random "'s, escape single '
            data = line.replace('"', "").replace("'", "''").strip().split(",")
            # Check if adress got split
            if len(data) > 20:
                data[13] = ",".join(data[13 : 13 + len(data) - 19])
            # Get needed columns, swap to list to allow edit to date column
            data = list(itemgetter(*USECOLS)(data))
            # Add "TO_DATE" to date data
            data[5] = f"TO_DATE('{data[5]}', 'dd/mm/yyyy')"
            # Add ' to str columns
            for col in range(7, len(data)):
                data[col] = f"'{data[col]}'"
            # Assemble line for file
            line_data = ", ".join(data)
            write_str = f"INSERT INTO ToyCarOrders VALUES ({line_data});"
            # Write to file
            write_file.write(write_str + "\n")
