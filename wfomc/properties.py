from collections import defaultdict
from aima3.utils import expr
from gmpy2 import mpz
from flint import fmpz, fmpq
from wfomc_cc import WFOMCWithCC
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import re
import os

def default_weight():
    return (mpz(1), mpz(1))

def no_divisor(n):
    return 1

def exp2(n): # Yes, this garbage IS needed for parallelism
    return 2**n

def exp6(n):
    return 6**n

def exp3(n):
    return 3**n

class Property:
    def __init__(self, name, formula, ccs = (), weights = (), increment=1, divisor=no_divisor):
        self.name = name
        self.formula = expr(re.sub(r'\s*', '', formula))
        self.ccs = ccs
        self.weights = defaultdict(default_weight, {key : (mpz(a), mpz(b)) for key, a, b in weights})
        self.increment = increment
        self.divisor = divisor

    def filename(self):
        return f"output/{self.name}.out"

    def parse_file(self):
        l = []

        filename = self.filename()
        if os.path.isfile(filename):
            with open(self.filename(), 'r') as f: # could use csv parser here instead
                for line in f.read().splitlines():
                    n, r, d = line.split(",")
                    l.append((int(n), int(r), float(d)))
        return l

    def last_uncomputed(self):
        computed = self.parse_file()
        if computed:
            return computed[-1][0] + 1
        else:
            return 0

    def evaluate(self, n):
        n *= self.increment
        ccs = [(k, v*n) for k,v in self.ccs]
        counter = WFOMCWithCC(self.formula, n, ccs=ccs)
        try:
            result = counter.get_wfomc(self.weights)
        except IndexError: # Actually a BUG in Tim's code when the polynomial is zero
            result = 0     # Simple fix as that can only happen when there is no model
        result = fmpq(result) / fmpq(self.divisor(n))
        return str(result) # Needed because fmpq isn't picklable

    def evaluate_time(self, n):
        start = datetime.now()
        result = self.evaluate(n)
        duration = (datetime.now() - start).total_seconds()
        return result, duration

    def evaluate_and_save(self, n):
        r, d = self.evaluate_time(n)
        with open(self.filename(), 'a') as out:
            print(f"{n},{r},{d}", file=out)
        return self.name, n, r, d

    def evaluate_and_saveuate_next(self):
        n = self.last_uncomputed()
        return self.evaluate_and_saveuate_next(n)

PROPERTIES = [
    Property("permutations",
        "(~F(x,y) | S1(x)) & (~F(x,y) | S2(y))",
        ccs = [('F', 1)],
        weights = [('S1', 1, -1), ('S2', 1, -1)]),
    Property("involutions",
        "(~F(x,y) | S1(x)) & (~F(x,y) | F(y,x))",
        ccs = [('F', 1)],
        weights = [('S1', 1, -1)]),
    Property("derangements",
        """~F(x,x) &
        (S1(x) | ~F1(x, y)) &
        (S2(x) | ~F2(y, x)) &
        (F1(x,y) | ~F(x,y)) &
        (~F1(x,y) | F(x,y)) &
        (F2(x,y) | ~F(x,y)) &
        (~F2(x,y) | F(x,y))""",
        ccs = [('F1', 1), ('F2', 1)],
        weights = [('S1', 1, -1), ('S2', 1, -1)]),
    Property("two_regularity",
        """~E(x, x) &
           (~E(x, y) | E(y, x)) &
           (~F(x, y) | E(x, y)) &
           (F(x, y) | ~E(x, y)) &
           (S1(x) | ~F1(x, y)) &
           (S2(x) | ~F2(x, y)) &
           (~F(x, y) | F1(x, y) | F2(x, y)) &
           (F(x, y) | ~F1(x, y)) &
           (F(x, y) | ~F2(x, y)) &
           (~F1(x, y) | ~F2(x, y))""",
        ccs = [('E', 2)],
        weights = [('S1', 1, -1), ('S2', 1, -1)],
        divisor = exp2),
    Property("three_regularity",
        """~E(x, x) &
        (~E(x, y) | E(y, x)) &
        (~F(x, y) | E(x, y)) &
        (F(x, y) | ~E(x, y)) &
        (S1(x) | ~F1(x, y)) &
        (S2(x) | ~F2(x, y)) &
        (S3(x) | ~F3(x, y)) &
        (~F(x, y) | F1(x, y) | F2(x, y) | F3(x, y)) &
        (F(x, y) | ~F1(x, y)) &
        (F(x, y) | ~F2(x, y)) &
        (F(x, y) | ~F3(x, y)) &
        (~F1(x, y) | ~F2(x, y)) &
        (~F1(x, y) | ~F3(x, y)) &
        (~F2(x, y) | ~F3(x, y))""",
        ccs = [('E', 3)],
        weights = [('S1', 1, -1), ('S2', 1, -1), ('S3', 1, -1)],
        increment = 2,
        divisor = exp6),
    Property("three_coloredness",
        """~E(x, x) &
        (~E(x, y) | E(y, x)) &
        (C1(x) | C2(x) | C3(x)) &
        (~C1(x) | ~C2(x)) &
        (~C2(x) | ~C3(x)) &
        (~C1(x) | ~C3(x)) &
        (~E(x,y) | (~(C1(x) & C1(y)) & ~(C2(x) & C2(y)) & ~(C3(x) & C3(y))))"""),
    Property("two_coloredness",
        """~E(x, x) &
        (~E(x, y) | E(y, x)) &
        (C1(x) | C2(x)) &
        (~C1(x) | ~C2(x)) &
        (~E(x,y) | (~(C1(x) & C1(y)) & ~(C2(x) & C2(y))))"""),
    Property("three_regularity_and_two_coloredness",
        """~E(x, x) &
        (~E(x, y) | E(y, x)) &
        (~F(x, y) | E(x, y)) &
        (F(x, y) | ~E(x, y)) &
        (S1(x) | ~F1(x, y)) &
        (S2(x) | ~F2(x, y)) &
        (S3(x) | ~F3(x, y)) &
        (~F(x, y) | F1(x, y) | F2(x, y) | F3(x, y)) &
        (F(x, y) | ~F1(x, y)) &
        (F(x, y) | ~F2(x, y)) &
        (F(x, y) | ~F3(x, y)) &
        (~F1(x, y) | ~F2(x, y)) &
        (~F1(x, y) | ~F3(x, y)) &
        (~F2(x, y) | ~F3(x, y)) &
        (C1(x) | C2(x)) &
        (~C1(x) | ~C2(x)) &
        (~E(x,y) | (~(C1(x) & C1(y)) & ~(C2(x) & C2(y))))""",
        ccs = [('E', 3)],
        increment = 2,
        weights = [('S1', 1, -1), ('S2', 1, -1), ('S3', 1, -1)],
        divisor = exp6),
    Property("two_regularity_and_three_coloredness",
        """~E(x, x) &
        (~E(x, y) | E(y, x)) &
        (~F(x, y) | E(x, y)) &
        (F(x, y) | ~E(x, y)) &
        (S1(x) | ~F1(x, y)) &
        (S2(x) | ~F2(x, y)) &
        (~F(x, y) | F1(x, y) | F2(x, y)) &
        (F(x, y) | ~F1(x, y)) &
        (F(x, y) | ~F2(x, y)) &
        (~F1(x, y) | ~F2(x, y)) &
        (C1(x) | C2(x) | C3(x)) &
        (~C1(x) | ~C2(x)) &
        (~C2(x) | ~C3(x)) &
        (~C1(x) | ~C3(x)) &
        (~E(x,y) | (~(C1(x) & C1(y)) & ~(C2(x) & C2(y)) & ~(C3(x) & C3(y))))""",
        ccs = [('E', 2)],
        weights = [('S1', 1, -1), ('S2', 1, -1)],
        divisor = exp2)
    ]

class Executor(ProcessPoolExecutor):
    def __init__(self, n, properties):
        ProcessPoolExecutor.__init__(self, n)
        self.properties = properties

    def submitProperty(self, p):
        future = self.submit(p.evaluate_next)
        future.add_done_callback(self.cb)

    def cb(self, future):
        p, r = future.result()
        print(f"{p.name} {r}")
        self.submitProperty(p)

def cb(fut):
    print(*fut.result())

with ProcessPoolExecutor(5) as executor:
    for n in range(100):
        for prop in PROPERTIES:
            future = executor.submit(prop.evaluate_and_save, n)
            future.add_done_callback(cb)
