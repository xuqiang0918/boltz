#!/usr/bin/env python3
"""Minimal reproducer: nn.Embedding / F.embedding is WRONG on SDAA when the
index tensor has rank >= 3.

Environment where this was observed
-----------------------------------
    container abb3-sdaa @ node499 (LoongArch, 10.71.15.115)
    PyTorch 2.7.1 | Torch-SDAA 3.2.0 | TecoDNN 3.2.0 | TecoBLAS 3.2.0
    device = /dev/tcaicard0   (the local `torch.cuda` reports unavailable)

No checkpoint, no dataset, no boltz import -- pure torch, ~2 seconds.

Result
------
    idx shape      expected                       SDAA actual
    (4,4)          all positions correct          correct        (rank 2 OK)
    (1,4,4)        all positions correct          only the FIRST 4 rows
    (2,4,4)        all positions correct          only the FIRST 8 rows
    (1,1,4,4)      all positions correct          only the FIRST 1-2 rows
                                                   (rank 4 broken too)
    (1,64,64)      all positions correct          only the FIRST 64 rows
    (1,117,117)    all positions correct          only the FIRST 117 rows

i.e. the kernel treats a rank-3 index (B, N, N) as (B*N, N) and writes only
B*N of the B*N*N output rows; the remaining rows keep whatever was in the
output buffer (they are NOT zero, and they contain NaN). The result is
reproducible within a process but differs between processes -- the classic
signature of uninitialised memory being read.

Reference used here is torch advanced indexing on the CPU (`W[idx]`), which
is exact, so there is no "both sides wrong the same way" blind spot.

Bypass that is correct on SDAA (all ranks):
    torch.index_select(W, 0, idx.reshape(-1)).reshape(*idx.shape, W.shape[-1])

USAGE
  python3 repro_embedding_sdaa.py
"""
import torch
import torch.nn.functional as F

ROWS, H = 7, 8


def devices():
    try:
        import torch.sdaa  # noqa: F401

        if torch.sdaa.is_available():
            return ["cpu", "sdaa"]
    except Exception as e:  # noqa: BLE001
        print("torch.sdaa unavailable (%r), running CPU self-check only" % (e,))
    return ["cpu"]


def main():
    torch.manual_seed(0)
    # a fixed, easily eyeballed table: row r is filled with r (scaled), so any
    # position that is wrong is obvious at a glance.
    W = (torch.arange(ROWS, dtype=torch.float32).reshape(-1, 1) * 100
         + torch.arange(H, dtype=torch.float32).reshape(1, -1))

    shapes = [(4, 4), (1, 4, 4), (2, 4, 4), (1, 1, 4, 4), (1, 64, 64),
              (1, 117, 117)]
    print("table (7,8): row r == [100r .. 100r+7]")
    print()
    print("%-16s %-8s %-22s %s" % ("idx.shape", "device", "wrong positions",
                                   "first wrong flat idx"))
    print("-" * 78)

    worst = {}
    for shape in shapes:
        idx = torch.randint(0, ROWS, shape)
        ref = W[idx]                      # CPU advanced indexing: exact
        for dev in devices():
            wi = W.to(dev)
            xi = idx.to(dev)
            try:
                out = F.embedding(xi, wi).cpu()
            except Exception as e:  # noqa: BLE001
                print("%-16s %-8s RAISED %s: %s" % (str(shape), dev,
                                                    type(e).__name__, e))
                continue
            # per-position comparison (a position is the H-vector at (.., i, j))
            wrong = (out != ref).reshape(-1, H).any(-1)
            nwrong = int(wrong.sum())
            first = int(wrong.nonzero()[0]) if nwrong else -1
            tot = wrong.numel()
            print("%-16s %-8s %-22s %s"
                  % (str(shape), dev, "%d/%d" % (nwrong, tot), first))
            worst[(shape, dev)] = nwrong

            if nwrong and nwrong > tot * 0.1:
                d = W  # cpu: `badvals` was already moved to cpu above
                badvals = out.reshape(-1, H)[wrong]
                matches_row = ((badvals[:, None, :] - d[None, :, :]).abs()
                               .sum(-1) == 0).any(1)
                print("%-16s %-8s   -> correct positions form a PREFIX of "
                      "length %d of %d" % ("", "", tot - nwrong, tot))
                print("%-16s %-8s      wrong rows all-zero: %d/%d ; wrong rows "
                      "equal to some table row: %d/%d"
                      % ("", "", int((badvals == 0).all(-1).sum()), nwrong,
                         int(matches_row.sum()), nwrong))
                out2 = F.embedding(xi, wi).cpu()
                print("%-16s %-8s      same call twice bitwise-identical: %s ; "
                      "NaN/inf present: %s"
                      % ("", "", bool(torch.equal(out, out2)),
                         bool((~torch.isfinite(out)).any())))

    print()
    print("=== bypass correctness (index_select) on every shape ===")
    print("%-16s %-8s %s" % ("idx.shape", "device", "matches CPU reference"))
    print("-" * 78)
    for shape in shapes:
        idx = torch.randint(0, ROWS, shape)
        ref = W[idx]
        for dev in devices():
            out = torch.index_select(W.to(dev), 0,
                                     idx.to(dev).reshape(-1)).reshape(*shape, H)
            print("%-16s %-8s %s" % (str(shape), dev,
                                     bool(torch.equal(out.cpu(), ref))))

    print()
    bad = [(k, v) for k, v in worst.items() if k[1] != "cpu" and v]
    if bad:
        print("CONCLUSION: SDAA nn.Embedding is WRONG for %d/%d tested shapes: %s"
              % (len(bad), len(shapes), sorted(s for s, _ in bad)))
    else:
        print("CONCLUSION: no failure observed on this build.")


if __name__ == "__main__":
    main()
