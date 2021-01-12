import os

directory = r'C:/MyFolder/Thesis Work/AFE-Imputation/Results - 2/'
out = ""
out += ",Missing-data percentage,,,,,,,,,,,,,,,,,,,Information about the data set,\n"
out += ",25(Complete),,,25(Incomplete),,,50(Complete),,,50(Complete),,,75(Complete),,,75(Complete),,,100(Complete)\n"
out += ",No Sampling,Random Sampling,LEU Sampling," \
       "No Sampling,Random Sampling,LEU Sampling," \
       "No Sampling,Random Sampling,LEU Sampling," \
       "No Sampling,Random Sampling,LEU Sampling," \
       "No Sampling,Random Sampling,LEU Sampling," \
       "No Sampling,Random Sampling,LEU Sampling," \
       ",Trivial Clf,Clf Type,Uncert Meas,Num Attrs,Type Attrs,Num Instances,Classes\n"

data_sets = {}

meas_map = {"conf": "confidence", "confi": "confidence*", "va": "variance", "var": "variance*", "e": "entropy",
            "en": "entropy*"}

for filename in os.listdir(directory):
    sp = filename.split("_")
    name = sp[0][8:]
    u_meas = meas_map[sp[-2]]
    clf = sp[-1]
    if name not in data_sets:
        data_sets[name] = ""

    f = open(directory + filename).read().split("\n")
    line1 = "Accuracies;"
    line2 = "AU-ROCs;"
    line3 = "Sampling Times;"

    for i in [9, 26, 44, 61, 79, 96]:
        for l in f[i: i + 3]:
            line1 += l.split("=")[-1] + ";"
        for l in f[i + 5:i + 8]:
            line2 += l.split("=")[-1] + ";"
        for l in f[i + 10:i + 13]:
            line3 += l.split("=")[-1] + ";"

    line1 += f[111].split("=")[1] + ";" + f[113].split("=")[1] + ";" + clf + ";" + u_meas + ";"
    line2 += f[112].split("=")[1]

    line1 += f[117].split("=")[1] + ";" + " {Cat: " + str(f[2].count("True")) + ", Real: " + str(
        f[2].count("False")) + "}; "  # num attrs
    line1 += f[116].split("=")[1] + ";" + f[115].split("(")[1][:-1]  # num instances and class counts

    o = line1 + "\n" + line2 + "\n" + line3[:-1] + "\n"

    o = o.replace(",", "~")
    o = o.replace(";", ",")
    o = o.replace("~", ";")

    data_sets[name] += o

for n, d in data_sets.items():
    f = open("tables2/" + n + ".csv", 'w')
    f.write(out)
    f.write(d)
    f.close()

# http://www.docsoso.com/excel/combine-excel.aspx
