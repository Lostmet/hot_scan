import os
import argparse
from collections import defaultdict
from pathlib import Path
from math import ceil
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import binom
from scipy.special import hyp2f1
import statsmodels.stats.multitest as sm


def load_chr_size(path: str) -> Tuple[int, Dict[str, int]]:
    """Load chromosome sizes from a file."""
    genome_size = 0
    size_by_chr: Dict[str, int] = {}
    with open(path, 'r') as fh:
        for line in fh:
            if not line.strip():
                continue
            chrom, size = line.split()[:2]
            if not size.isdigit():
                continue
            size = int(size)
            size_by_chr[chrom] = size
            genome_size += size
    return genome_size, size_by_chr


def load_sample(path: str) -> Tuple[int, Dict[str, List[Tuple[str, int, int]]]]:
    """Load events grouped by chromosome."""
    events_by_chr: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    count = 0
    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith('chr'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[:3]
            events_by_chr[chrom].append((chrom, int(start), int(end)))
            count += 1
    return count, events_by_chr


def ss_pvalue(ss_N: int, ss_k: int, ss_G: float, ss_a: int, ss_m: int) -> float:
    density_bin = binom.pmf(ss_k, ss_a, ss_G)
    try:
        hyper = hyp2f1(1, ss_k - ss_a, 1 + ss_k, ss_G / (ss_G - 1))
    except Exception:
        hyper = 0.0
    return (ss_k * ss_N / ss_m - ss_a) * density_bin + 2 * density_bin * hyper


def mk_hash_table(ss_a: int, ss_m: int, ss_N: int) -> Dict[int, float]:
    ss_G = ss_m / ss_N
    return {k: ss_pvalue(ss_N, k, ss_G, ss_a, ss_m) for k in range(1, ss_m + 1)}


def look_for_islands(events: List[Tuple[str, int, int]], ss_m: int) -> Tuple[Dict[int, List[int]], Dict[int, List[str]]]:
    islands_pos: Dict[int, List[int]] = defaultdict(list)
    islands_all: Dict[int, List[str]] = defaultdict(list)
    last_start = None
    island_id = 1
    for chrom, start, end in sorted(events, key=lambda x: x[1]):
        info = f"{chrom}\t{start}\t{end}\n"
        islands_pos[island_id].append(start)
        islands_all[island_id].append(info)
        if last_start is not None:
            if start - last_start + 1 > ss_m:
                island_id += 1
        last_start = start
    return islands_pos, islands_all


def calc_ss_pvalue(islands_pos: Dict[int, List[int]], islands_all: Dict[int, List[str]],
                    hash_table: Dict[int, float], ss_m: int, chrom: str,
                    ss_a: int, ss_N: int, sample: str, out_dir: str, adjust: str):
    ss_G = ss_m / ss_N
    chrom_dir = Path(out_dir) / sample / str(ss_m)
    chrom_dir.mkdir(parents=True, exist_ok=True)
    out_log = chrom_dir / f"{chrom}.log"
    out_txt = chrom_dir / f"{chrom}.txt"

    extend = ss_m // 2
    with open(out_log, 'w') as log_fh, open(out_txt, 'w') as txt_fh:
        for island_id in sorted(islands_pos):
            positions = islands_pos[island_id]
            info = islands_all[island_id]
            start_i = positions[0]
            end_i = positions[-1]
            size = end_i - start_i + 1
            status = 'O'
            if end_i - start_i < ss_m:
                middle = (start_i + end_i) // 2
                start_i = middle - extend
                end_i = middle + extend
                status = 'U'
            log_fh.write(f"\n{island_id}|{status}|{size}|{chrom}:{start_i}-{end_i}\t" + "-".join(map(str, positions)) + "\n")
            log_fh.write("{}|".format(island_id) + "{}".format("{}".join(info)))
            for x in range(start_i, end_i, ss_m):
                c = x + ss_m
                for inc in range(ss_m):
                    start = x + inc
                    end = start + ss_m
                    if end > end_i:
                        break
                    k = 0
                    pos = start
                    while pos <= end:
                        if pos in positions:
                            k += 1
                        pos += 1
                    if k not in hash_table:
                        hash_table[k] = ss_pvalue(ss_N, k, ss_G, ss_a, ss_m)
                    txt_fh.write(f"{chrom}\t{start}\t{end}\t{k}\t{island_id}\t{hash_table[k]}\n")

    if adjust == 'BY':
        # adjust p-values using Benjamini-Yekutieli procedure
        data = []
        with open(out_txt) as fh:
            for row in fh:
                parts = row.strip().split()  # chrom start end k id p
                data.append(parts)
        if data:
            pvals = [float(d[5]) for d in data]
            _, p_adj, _, _ = sm.multipletests(pvals, method='fdr_by')
            by_dir = chrom_dir / 'BY'
            by_dir.mkdir(exist_ok=True)
            with open(by_dir / f"{chrom}.txt", 'w') as fh:
                for d, p in zip(data, p_adj):
                    if 0 <= p < 0.05:
                        fh.write("{} {} {} {} {} {}\n".format(*d[:5], p))


def calc_ss_by_cg(size_by_chr: Dict[str, int], events_by_chr: Dict[str, List[Tuple[str, int, int]]],
                   ss_m: int, sample: str, adjust: str, out_dir: str):
    Path(out_dir, sample).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir, sample, 'params.txt'), 'w') as param_fh:
        for chrom, events in events_by_chr.items():
            ss_N = size_by_chr[chrom]
            ss_a = len(events)
            hash_table = mk_hash_table(ss_a, ss_m, ss_N)
            param_fh.write(f"{chrom}\t{ss_N}\t{ss_a}\n")
            islands_pos, islands_all = look_for_islands(events, ss_m)
            calc_ss_pvalue(islands_pos, islands_all, hash_table, ss_m, chrom,
                           ss_a, ss_N, sample, out_dir, adjust)


def parse_args():
    parser = argparse.ArgumentParser(description="Scan statistics for hotspot detection")
    parser.add_argument('-m', required=True, type=int, help='window width m')
    parser.add_argument('-c', required=True, help='chromosome size file')
    parser.add_argument('-e', required=True, help='events file')
    parser.add_argument('-o', required=True, help='output directory')
    parser.add_argument('-s', type=float, default=0.05, help='significance level')
    parser.add_argument('-a', choices=['yes', 'no'], default='no', help='adjust p-values BY')
    return parser.parse_args()


def main():
    args = parse_args()
    adjust = 'BY' if args.a == 'yes' else 'un'
    genome_size, size_by_chr = load_chr_size(args.c)
    events_in_genome, events_by_chr = load_sample(args.e)
    calc_ss_by_cg(size_by_chr, events_by_chr, args.m, args.e, adjust, os.path.join(args.o, 'scan_out'))


if __name__ == '__main__':
    main()
