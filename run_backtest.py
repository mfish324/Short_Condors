#!/usr/bin/env python
"""
Command-line tool to run 0DTE options backtests.

Usage:
    python run_backtest.py --start 2024-10-01 --end 2024-12-31 --delta 0.10
    python run_backtest.py --start 2024-06-01 --end 2024-12-31 --delta 0.05 --times 10:00,11:00
"""
import argparse
import json
from datetime import datetime

from backtest_v2 import run_backtest, print_results


def parse_times(times_str: str) -> list[tuple[int, int]]:
    """Parse comma-separated time strings like '10:00,10:30,11:00'"""
    times = []
    for t in times_str.split(','):
        t = t.strip()
        hour, minute = t.split(':')
        times.append((int(hour), int(minute)))
    return times


def main():
    parser = argparse.ArgumentParser(
        description='Backtest 0DTE options selling strategy'
    )
    parser.add_argument(
        '--start', '-s',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', '-e',
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--delta', '-d',
        type=float,
        default=0.10,
        help='Target delta (default: 0.10)'
    )
    parser.add_argument(
        '--times', '-t',
        default='10:00,10:30,11:00',
        help='Entry times ET, comma-separated (default: 10:00,10:30,11:00)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed trade-by-trade output'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
        datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print('Error: Dates must be in YYYY-MM-DD format')
        return 1

    entry_times = parse_times(args.times)

    print(f'0DTE Options Backtest')
    print(f'=' * 60)
    print(f'Date range: {args.start} to {args.end}')
    print(f'Target delta: {args.delta}')
    print(f'Entry times (ET): {args.times}')
    print(f'=' * 60)

    result = run_backtest(
        start_date=args.start,
        end_date=args.end,
        entry_times=entry_times,
        target_delta=args.delta,
        verbose=args.verbose
    )

    print_results(result)

    if args.output:
        output_data = {
            'parameters': {
                'start_date': args.start,
                'end_date': args.end,
                'target_delta': args.delta,
                'entry_times': args.times
            },
            'results': {
                'total_trades': result.total_trades,
                'otm_count': result.otm_count,
                'itm_count': result.itm_count,
                'otm_probability': result.otm_probability,
                'avg_premium_collected': result.avg_premium_collected,
                'total_pnl': result.total_pnl,
                'avg_delta': result.avg_delta
            },
            'trades': result.trades
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nResults saved to {args.output}')

    return 0


if __name__ == '__main__':
    exit(main())
