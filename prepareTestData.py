import os
from shutil import copyfile

#POSITIVE
counter = 0
for case_name in ["bark", "bikes", "boat", "graf", "leuven", "trees", "wall"]:
    file = open("samples/"+case_name+"/matches.csv")
    lines = file.readlines()
    step = int((len(lines) - 3 )/ 50)
    start = 0
    for x in range(50):
        nums = lines[start].split(",")

        xstr = nums[0].__str__()
        while len(xstr) < 5:
            xstr = "0" + xstr
        xstr += ".png"
        path_from = "samples/"+case_name.__str__()+"/"+xstr
        path_to = "test/positive-p"+counter.__str__()+"-1.png"
        copyfile(path_from, path_to)

        xstr = nums[1].__str__().rstrip()
        while len(xstr) < 5:
            xstr = "0" + xstr
        xstr += ".png"
        path_from = "samples/"+case_name.__str__()+"/"+xstr
        path_to = "test/positive-p"+counter.__str__()+"-2.png"
        copyfile(path_from, path_to)

        start += step
        counter += 1

#NEGATIVE
counter = 0
for case_name in ["bark", "bikes", "boat", "graf", "leuven", "trees", "wall"]:

    list = os.listdir("samples/"+case_name)
    file_count = len(list) - 3
    step = file_count / 2
    if step > 25:
        step = int(step / 30)
    start = 0
    for x in range(50):
        xstr = (start).__str__()
        while len(xstr) < 5:
            xstr = "0" + xstr
        xstr += ".png"
        path_from = "samples/"+case_name.__str__()+"/"+xstr
        path_to = "test/negative-p"+counter.__str__()+"-1.png"
        copyfile(path_from, path_to)

        xstr = (file_count - start).__str__()
        while len(xstr) < 5:
            xstr = "0" + xstr
        xstr += ".png"
        path_from = "samples/"+case_name.__str__()+"/"+xstr
        path_to = "test/negative-p"+counter.__str__()+"-2.png"
        copyfile(path_from, path_to)

        start += step
        counter += 1