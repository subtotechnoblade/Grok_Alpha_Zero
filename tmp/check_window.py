"""Which generations land in the training window, old logic vs new. No files needed."""


def old(gen, num_prev=4):
    files = list(range(gen + 1))
    files = files[-num_prev - 1:]
    if gen >= 2:
        files = files[1:]
    return files


def new(gen, num_prev=4):
    files = list(range(gen + 1))
    if gen >= 1:
        files = [f for f in files if f != 0]
    return files[-num_prev - 1:]


print(f"{'training gen':>12} | {'OLD window':<22} | {'NEW window':<22}")
print("-" * 64)
for g in range(0, 11):
    o, n = old(g), new(g)
    flag = "" if o == n else "   <-- changed"
    print(f"{g:>12} | {str(o):<22} | {str(n):<22}{flag}")
