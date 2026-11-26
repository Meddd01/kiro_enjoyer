from fractions import Fraction
import json
from math import ceil


def load_instance(path):
    """Read an official instance without changing its public schema."""
    with open(path, encoding="utf-8-sig") as stream:
        return json.load(stream)


class Problem:
    def __init__(self, instance):
        self.flows = instance["flows"]
        self.routes = instance["routes"]
        self.carriers = instance["carriers"]
        self.ports = instance["ports"]
        self.sites = instance["sites"]
        self.scenarios = instance["scenarios"]
        self.spot = next((c["id"] for c in self.carriers if c["name"] == "SPOT"), None)
        self.spot_port = next((p["id"] for p in self.ports if p["code"] == "SPOT"), None)
        alpha = Fraction(str(instance["meta"]["alpha"]))
        self.minimum = [
            0 if c["id"] == self.spot else ceil(alpha * c["trans"])
            for c in self.carriers
        ]
        self.failed = [frozenset(s["routes"]) for s in self.scenarios]
        self.nominal = self.failed.index(frozenset())
        self.mask = [0] * len(self.routes)
        for s, failed in enumerate(self.failed):
            for r in failed:
                self.mask[r] |= 1 << s

        truck = [{p["port"]: p["cost"] for p in t["ports"]} for t in self.sites]
        self.ends = []
        self.truck_in = []
        self.truck_out = []
        self.cost = []
        self.capacity = []
        for route in self.routes:
            flow = self.flows[route["flow"]]
            origin, dest = flow["t_dep"], flow["t_arr"]
            depart, arrive = route["p_dep"], route["p_arr"]
            send, receive = truck[origin][depart], truck[dest][arrive]
            self.ends.append((origin, dest))
            self.truck_in.append(receive)
            self.truck_out.append(send)
            self.cost.append(2 * (send + receive) + route["cost"])
            self.capacity.append(min(
                self.carriers[route["carrier"]]["trans"],
                self.ports[depart]["manut"] // (1 + (depart == arrive)),
                self.ports[arrive]["manut"],
            ))

    def spot_solution(self):
        quantities = {}
        for f in self.flows:
            if not f["n"]:
                continue
            r = next((
                r for r in f["routes"]
                if self.spot is not None and self.spot_port is not None
                and self.routes[r]["carrier"] == self.spot and not self.mask[r]
                and self.routes[r]["p_dep"] == self.routes[r]["p_arr"] == self.spot_port
            ), None)
            if r is None:
                return None
            quantities[r] = f["n"]
        return pack(quantities, {})


def pack(x, y):
    return {
        "x": [{"route": r, "n": n} for r, n in sorted(x.items()) if n > 0],
        "y": [
            {"from": i, "to": j, "n": n}
            for (i, j), n in sorted(y.items()) if n > 0
        ],
    }


def unpack(solution):
    values = []
    for key, names in (("x", ("route",)), ("y", ("from", "to"))):
        entries = {}
        for row in solution[key]:
            ids = tuple(row[name] for name in names)
            n = row["n"]
            if any(type(i) is not int or i < 0 for i in ids) or type(n) is not int or n <= 0:
                raise ValueError("Expected nonnegative integer IDs and positive integer quantities")
            index = ids[0] if key == "x" else ids
            if index in entries:
                raise ValueError("Duplicate solution entry")
            entries[index] = n
        values.append(entries)
    return tuple(values)
