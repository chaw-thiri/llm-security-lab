"""Produce complete per-seed and aggregate contextual study results."""
from promptshield.reporting import write_report as write_compact_report


def main():
    write_compact_report("context-v2")


if __name__ == '__main__':
    main()
