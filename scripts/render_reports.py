"""Render combined compact reports from saved JSON; no training, downloads or inference."""
import argparse
from promptshield.reporting import write_all_reports


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    write_all_reports(check=args.check)
    print("Compact study reports verified" if args.check else "Compact study reports generated")
