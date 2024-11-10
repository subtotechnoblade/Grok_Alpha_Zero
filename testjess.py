def _check_row(row):
    if row[0] != 0 and row[0] == row[1] == row[2]:
        return row[0]
    return None



if __name__ == "__main__":
    test_row = [1, 1, 1]
    print (_check_row(test_row))
