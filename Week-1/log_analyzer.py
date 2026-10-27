import sys
import csv


def analyze_logs(file_path):
    errors = {}

    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()

                if line.startswith("ERROR:"):
                    error_message = line.replace("ERROR:", "", 1).strip()
                    errors[error_message] = errors.get(error_message, 0) + 1

    except FileNotFoundError:
        print("Error: Input file not found.")
        return None

    return errors


def export_csv(errors):
    with open("error_summary.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Error Message", "Count"])

        for error, count in errors.items():
            writer.writerow([error, count])

    print("Error summary exported to error_summary.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <log_file_path>")
        return

    file_path = sys.argv[1]

    errors = analyze_logs(file_path)

    if errors is not None:
        export_csv(errors)


if __name__ == "__main__":
    main()