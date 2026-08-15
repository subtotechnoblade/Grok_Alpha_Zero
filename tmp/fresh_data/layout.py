import h5py as h5
import numpy as np

p = "Connect4/Grok_Zero_Train/9/Self_Play_Data.h5"
with h5.File(p, "r") as f:
    keys = list(f.keys())
    print("num keys:", len(keys))
    print("sample keys:", sorted(keys)[:8])
    print("game_stats:", f["game_stats"][:])
    print("boards_0:", f["boards_0"].shape, f["boards_0"].dtype)
    print("policies_0:", f["policies_0"].shape, f["policies_0"].dtype)
    print("values_0:", f["values_0"].shape, f["values_0"].dtype)
    print("boards_0[0] channels (transposed to c,h,w):")
    b = f["boards_0"][0]
    print(b.transpose(2, 0, 1))
    print("policies_0[0]:", f["policies_0"][0])
    print("policies_1[0]:", f["policies_1"][0])
    print("values_0[:5].ravel():", f["values_0"][:5].ravel())
    print("values_1[:5].ravel():", f["values_1"][:5].ravel())
    ntrip = (len(keys) - 1) // 3
    print("dataset triples:", ntrip, "games:", int(f["game_stats"][2]))
