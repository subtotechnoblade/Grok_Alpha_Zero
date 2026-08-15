"""M11: one table. Raw per-sample objective, MSE against pure z, gen N data / model N."""
import json, re
rows = json.load(open("m3_net.json"))
gbm = {}
for blk in open("m6_out.txt", encoding="utf-8").read().split("=== ")[1:]:
    m = re.match(r"(run\d) gen (\d+)", blk)
    rz = re.search(r"target=z:\s+RAW held-out MSE = ([\d.]+)", blk)
    rt = re.search(r"target=t:\s+RAW held-out MSE = ([\d.]+)", blk)
    if m and rz:
        gbm[(m.group(1), int(m.group(2)))] = (float(rz.group(1)), float(rt.group(1)))

print("RAW per-sample MSE against pure z, evaluated on generation N (held out for model N)")
print(f"{'run':>5} {'gen':>4} | {'const-0':>8} {'floor(rep)':>10} {'q-bound':>8} {'GBM':>7} {'ResNet':>7} "
      f"| {'ResNet/GBM':>10} {'ResNet/const0':>13}")
for r in rows:
    if r["bucket"] != "ALL" or r["cb"] != "any":
        continue
    g = gbm.get((r["run"], r["gen"]), (float('nan'),)*2)
    print(f"{r['run']:>5} {r['gen']:>4} | {r['const0']:>8.4f} {r['floor']:>10.4f} {r['qbound']:>8.4f} "
          f"{g[0]:>7.4f} {r['mse_net_z']:>7.4f} | {r['mse_net_z']/g[0]:>10.2f} "
          f"{r['mse_net_z']/r['const0']:>13.2f}")

print("\nSame, MSE against the STORED TRAINING TARGET t (what the loss actually optimises)")
print(f"{'run':>5} {'gen':>4} | {'GBM(t)':>8} {'ResNet(t)':>9} | {'ResNet/GBM':>10}")
for r in rows:
    if r["bucket"] != "ALL" or r["cb"] != "any":
        continue
    g = gbm.get((r["run"], r["gen"]), (float('nan'),)*2)
    print(f"{r['run']:>5} {r['gen']:>4} | {g[1]:>8.4f} {r['mse_net_t']:>9.4f} | {r['mse_net_t']/g[1]:>10.2f}")

print("\nBy ply bucket, gen 10 and 15, MSE against z: floor / GBM / ResNet")
print(f"{'run':>5} {'gen':>4} {'bucket':>7} | {'const-0':>8} {'floor':>7} {'q-bnd':>7} {'ResNet':>7} {'ratio n/f':>9}")
for r in rows:
    if r["cb"] != "any" or r["gen"] not in (10, 15) or r["bucket"] == "ALL":
        continue
    print(f"{r['run']:>5} {r['gen']:>4} {r['bucket']:>7} | {r['const0']:>8.4f} {r['floor']:>7.4f} "
          f"{r['qbound']:>7.4f} {r['mse_net_z']:>7.4f} {r['mse_net_z']/r['floor']:>9.1f}")
