import argparse
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    path = args.path

    names = set([str(f).split(".")[0]
                 for f in listdir(path) if isfile(join(path, f)) and "log" not in f and "fc" not in f])
    c1 = [" --data " + str(n) for n in names]
    c2 = [" --um " + um for um in ["confidence", "variance", "entropy", "confidence*", "variance*", "entropy*"]]
    c3 = [" --clf " + clf for clf in ["nn", "dt"]]
    commands = "\n".join(["python Main.py" + a + b + c + " &" for a in c1 for b in c2 for c in c3])
    f = open("run_all", 'w')
    f.write(commands)
