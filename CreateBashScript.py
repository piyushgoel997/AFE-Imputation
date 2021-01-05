import argparse
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="data")
    args = parser.parse_args()
    path = args.path

    c1 = set([str(f).split(".")[0] for f in listdir(path) if isfile(join(path, f)) and "_cat" not in f])
    c2 = ["confidence", "variance", "entropy", "confidence*", "variance*", "entropy*"]
    c3 = ["nn", "dt"]
    names = []
    for a in c1:
        for b in c2:
            for c in c3:
                n = a + "_" + b[:-6] + "_" + c
                # mem = "10G"
                mem = "100G"
                if "adult" in n or "avila" in n or "bank" in n:
                    # mem = "100G"
                    mem = "200G"
                # if "frogs" in n or "mushroom" in n or "shill" in n or "wine" in n or "spam" in n:
                #     mem = "20G"
                command = "#!/bin/bash\n" \
                          "#SBATCH --time=24:00:00\n" \
                          "#SBATCH -n 10\n" \
                          "#SBATCH --job-name=" + n + "\n" \
                          "#SBATCH --partition=short\n" \
                          "#SBATCH --mem=" + mem + "\n" \
                          "#SBATCH --mail-user=goel.pi@northeastern.edu\n" \
                          "#SBATCH --mail-type=END\n" \
                          "#SBATCH -o job_logs/" + n + "-%j\n" \
                          "\n" \
                          "module purge\n" \
                          "module load python/3.7.1 anaconda3/3.7\n" \
                          "\n" \
                          "srun python Main.py" + " --data " + a + " --um " + b + " --clf " + c

                f = open("scripts/" + n, 'w')
                names.append("scripts/" + n)
                f.write(command)

    f = open("run_all", 'w')
    f.write("sbatch ")
    f.write("\nsbatch ".join(names))
