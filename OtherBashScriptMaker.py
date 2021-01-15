import os
import shutil

if __name__ == "__main__":

    try:
        shutil.rmtree("other_scripts")
    except:
        print()

    os.mkdir("other_scripts")
    os.mkdir("other_scripts/individual")

    try:
        os.mkdir("other_logs")
    except:
        print()

    try:
        os.mkdir("other_saved_results")
    except:
        print()

    memory = {"abalone_0": "10G", "adult_0": "100G", "avila_0": "100G", "bank_0": "100G", "biodeg_0": "10G",
              "cardiotocography_0": "20G", "car_0": "5G", "credit_card_defaulters_0": "100G",
              "drug_consumption_5": "10G", "faults_0": "20G", "frogs_0": "50G", "mushroom_0": "20G",
              "obesity_0": "10G", "online_shoppers_intention_0": "100G", "phishing_0": "5G", "sat_0": "50G",
              "Sensorless_drive_diagnosis_0": "500G", "shill_bidding_0": "20G", "spambase_0": "100G",
              "statlog-is_0": "10G", "statlog-ls_0": "20G", "wine_quality_merged_0": "20G"}

    num_features = [8, 13, 10, 16, 41, 25, 6, 23, 12, 27, 20, 22, 16, 17, 9, 36, 100, 9, 57, 19, 36, 12]

    c2 = ["confidence", "variance", "entropy"]
    c3 = ["nn", "rf"]
    names = []
    partition = "short"
    time = "24:00:00"
    for a, mem in [x for _, x in sorted(zip(num_features, memory.items()), key=lambda item: item[0])]:
        for b in c2:
            for c in c3:
                if "frogs" in a:
                    partition = "long"
                    time = "5-00:00"
                n = a + "_" + b[:-6] + "_" + c
                command = "#!/bin/bash\n" \
                          "#SBATCH --time=" + time + "\n" \
                          "#SBATCH -c 10\n" \
                          "#SBATCH --job-name=other_" + n + "\n" \
                          "#SBATCH --partition=" + partition + "\n" \
                          "#SBATCH --mem=" + mem + "\n" \
                          "#SBATCH --mail-user=goel.pi@northeastern.edu\n" \
                          "#SBATCH --mail-type=END\n" \
                          "#SBATCH -o other_logs/%j" + n + "\n" \
                          "\n" \
                          "module purge\n" \
                          "module load python/3.7.1 anaconda3/3.7\n" \
                          "\n" \
                          "srun python ContinuousSampling.py" + " --data " + a + " --um " + b + " --clf " + c

                f = open("other_scripts/individual/" + n, 'w')
                names.append("other_scripts/individual/" + n)
                f.write(command)

    for i in range(0, len(names), 6):
        f = open("other_scripts/run_all_" + str(int(i / 6) + 1), 'w')
        f.write("sbatch ")
        f.write("\nsbatch ".join(names[i: i + 6]))


# from os import listdir
# from os.path import isfile, join
#
# c2 = ["confidence", "variance", "entropy"]
# c3 = ["nn", "rf"]
# path = "data"
# names = set([str(f).split(".")[0]
#              for f in listdir(path) if isfile(join(path, f)) and "_cat." not in f])
# commands = []
# for a in names:
#     for b in c2:
#         for c in c3:
#             commands.append("echo \"" + a + "\"\npython ContinuousSampling.py" + " --data " + a + " --um " + b + " --clf " + c)
# f = open("run_all", 'w')
# f.write("\n".join(commands))
