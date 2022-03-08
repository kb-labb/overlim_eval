import os
from tabulate import tabulate  # type: ignore
from typing import Dict, Optional, Any

DoubleDict = Dict[str, Dict[str, float]]

tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst", "stsb", "wnli", "boolq", "cb"]


def get_results(logfile: str) -> DoubleDict:
    results: DoubleDict = {}
    with open(logfile) as fh:
        prefix = None
        for line in fh:
            if line.startswith("***** eval metrics *****"):
                prefix = "dev"
                results[prefix] = {}
            elif line.startswith("***** predict metrics *****"):
                prefix = "test"
                results[prefix] = {}
            elif prefix:
                _line = line.split()
                if len(_line) == 3 and _line[1] == "=":
                    _category, _, _value = _line
                    if _category.startswith("eval"):
                        category = _category.split("_")[1]
                        if category in ["accuracy", "f1", "pearson", "spearmanr"]:
                            value = float(_value)
                            results[prefix][category] = value
                else:
                    prefix = None
    return results


def get_best_results(runs: Dict[str, DoubleDict]):
    official_best_score = -float("inf")
    official_best_run = None
    best_dev_score = -float("inf")
    best_dev_run = None
    best_test_score = -float("inf")
    best_test_run = None
    avg_dev = 0.0
    avg_test = 0.0

    num_runs = len(runs)
    for run, results in runs.items():
        if not ("dev" in results and "test" in results):
            num_runs -= 1
            continue
        # average for multiple measures
        dev_score = sum(results["dev"].values()) / len(results["dev"])
        test_score = sum(results["test"].values()) / len(results["test"])
        avg_dev += dev_score
        avg_test += test_score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_dev_run = run
        if test_score > best_test_score:
            best_test_score = test_score
            best_test_run = run
    if best_dev_run is None or best_test_run is None:
        return None, None, None, None, None, num_runs

    official_best_run = best_dev_run
    official_best_score = sum(runs[best_dev_run]["test"].values()) / len(runs[best_dev_run]["test"].values())

    avg_dev /= num_runs
    avg_test /= num_runs

    return (official_best_run, official_best_score), (best_dev_run, best_dev_score), (best_test_run, best_test_score), avg_dev, avg_test, num_runs


def main() -> None:
    files = os.listdir("logs/")
    results: Dict[str, Dict[str, DoubleDict]] = {}
    for fn in files:
        model, task, run, _ = fn.split(".")
        if task not in results:
            results[task] = {}
        if model not in results[task]:
            results[task][model] = {}
        results[task][model][run] = get_results(os.path.join("logs", fn))

    tables: Dict[str, Dict[str, Dict[str, Optional[Any]]]] = {"avg_dev": {}, "avg_test": {}, "max_dev": {}, "max_test": {}, "official": {}, "off_avg": {}}
    for task in tasks:
        for model in results[task]:
            official, dev, test, avg_dev, avg_test, num_runs = get_best_results(results[task][model])
            if model not in tables["official"]:
                for k in tables:
                    tables[k][model] = {"Model": model}
                # tables[model] = {"Model": model}
            if task not in tables["official"][model]:
                for k in tables:
                    tables[k][model][task] = {}
            tables["avg_dev"][model][task] = avg_dev
            tables["avg_test"][model][task] = avg_test
            tables["max_dev"][model][task] = dev[1] if dev is not None else None
            tables["max_test"][model][task] = test[1] if test is not None else None
            if official is not None:
                official = results[task][model][official[0]]["test"]
                tables["official"][model][task] = tuple(official.values()) if len(official.values()) > 1 else tuple(official.values())[0]
                tables["off_avg"][model][task] = sum(official.values())/len(official.values()) if len(official.values()) > 1 else tuple(official.values())[0]
                # table[model][task] = official[1]
            else:
                tables["official"][model][task] = None
                tables["off_avg"][model][task] = None

    fmt = "github"
    with open("results.md", "w") as fout:
        print("avg_dev", file=fout)
        print(tabulate(tables["avg_dev"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)
        print("", file=fout)

        print("max_dev", file=fout)
        print(tabulate(tables["max_dev"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)
        print("", file=fout)

        print("avg_test", file=fout)
        print(tabulate(tables["avg_test"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)
        print("", file=fout)

        print("max_test", file=fout)
        print(tabulate(tables["max_test"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)
        print("", file=fout)

        print("off_avg", file=fout)
        print(tabulate(tables["off_avg"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)
        print("", file=fout)

        print("official", file=fout)
        print(tabulate(tables["official"].values(), headers="keys", tablefmt=fmt, floatfmt=".2%"), file=fout)


if __name__ == "__main__":
    main()
