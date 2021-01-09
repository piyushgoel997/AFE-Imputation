import argparse
from os import listdir
from os.path import isfile, join, getsize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="data")
    args = parser.parse_args()
    path = args.path
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and "_cat" not in f]
    c1 = [str(f).split("/")[-1].split("\\")[-1].split(".")[0] for f in sorted(files, key=getsize)]
    c2 = ["confidence", "variance", "entropy", "confidence*", "variance*", "entropy*"]
    c3 = ["nn", "rf"]
    names = []
    mem = "200G"
    for a in c1:
        for b in c2:
            for c in c3:
                n = a + "_" + b[:-6] + "_" + c
                if "spam" in n:
                    mem = "500G"
                command = "#!/bin/bash\n" \
                          "#SBATCH --time=24:00:00\n" \
                          "#SBATCH -c 10\n" \
                          "#SBATCH --job-name=" + n + "\n" \
                          "#SBATCH --partition=short\n" \
                          "#SBATCH --mem=" + mem + "\n" \
                          "#SBATCH --mail-user=goel.pi@northeastern.edu\n" \
                          "#SBATCH --mail-type=END\n" \
                          "#SBATCH -o job_logs/%j" + n + "\n" \
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
