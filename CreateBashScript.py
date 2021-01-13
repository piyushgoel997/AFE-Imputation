import os
import shutil

if __name__ == "__main__":

    try:
        shutil.rmtree("scripts")
    except:
        print()

    os.mkdir("scripts")
    os.mkdir("scripts/individual")

    try:
        os.mkdir("logs")
    except:
        print()

    try:
        os.mkdir("saved_results")
    except:
        print()

    memory = {"abalone_0": "100G", "adult_0": "500G", "avila_0": "500G", "bank_0": "500G", "biodeg_0": "100G",
              "cardiotocography_0": "200G", "car_0": "50G", "credit_card_defaulters_0": "500G",
              "drug_consumption_5": "100G", "faults_0": "200G", "frogs_0": "500G", "mushroom_0": "200G",
              "obesity_0": "100G", "online_shoppers_intention_0": "500G", "phishing_0": "50G", "sat_0": "500G",
              "Sensorless_drive_diagnosis_0": "5000G", "shill_bidding_0": "200G", "spambase_0": "1000G",
              "statlog-is_0": "100G", "statlog-ls_0": "200G", "wine_quality_merged_0": "200G"}
    num_features = [8, 13, 10, 16, 41, 25, 6, 23, 12, 27, 20, 22, 16, 17, 9, 36, 100, 9, 57, 19, 36, 12]

    c2 = ["confidence", "variance", "entropy"]
    c3 = ["nn", "rf"]
    names = []
    partition = "short"
    time = "24:00:00"
    for a, mem in [x for _, x in sorted(zip(num_features, memory.items()), key=lambda item: item[0])]:
        for b in c2:
            for c in c3:
                if "sat" in a:
                    partition = "long"
                    time = "5-00:00"
                n = a + "_" + b[:-6] + "_" + c
                command = "#!/bin/bash\n" \
                          "#SBATCH --time=" + time + "\n" \
                          "#SBATCH -c 10\n" \
                          "#SBATCH --job-name=" + n + "\n" \
                          "#SBATCH --partition=" + partition + "\n" \
                          "#SBATCH --mem=" + mem + "\n" \
                          "#SBATCH --mail-user=goel.pi@northeastern.edu\n" \
                          "#SBATCH --mail-type=END\n" \
                          "#SBATCH -o logs/%j" + n + "\n" \
                          "\n" \
                          "module purge\n" \
                          "module load python/3.7.1 anaconda3/3.7\n" \
                          "\n" \
                          "srun python Main.py" + " --data " + a + " --um " + b + " --clf " + c

                f = open("scripts/individual/" + n, 'w')
                names.append("scripts/individual/" + n)
                f.write(command)

    for i in range(0, len(names), 6):
        f = open("scripts/run_all_" + str(int(i / 6) + 1), 'w')
        f.write("sbatch ")
        f.write("\nsbatch ".join(names[i: i + 6]))
