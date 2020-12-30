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
    commands = "\n".join(["python Main.py --data " + n + " &" for n in names])
    f = open("run_all", 'w')
    f.write(commands)
