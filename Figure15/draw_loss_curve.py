import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 16,
    "mathtext.fontset":'stix',
}
rcParams.update(config)

font1 = {
    'family': 'DejaVu Sans',
    'weight': 'semibold',
    'style':'normal',
    'size': 20
}


# if not os.path.exists('loss_record/loss.txt'):
#     print("data not exists, generating loss.txt")

DLRM_list = []
TT_Rec_list = []
EL_Rec_list = []

DLRM_file = open('loss_record/DLRM.txt')
TT_Rec_file = open('loss_record/TTRec.txt')
EL_Rec_file = open('loss_record/ELRec.txt')

for line in DLRM_file:
    DLRM_list.append(float(line))
for line in TT_Rec_file:
    TT_Rec_list.append(float(line))
for line in EL_Rec_file:
    EL_Rec_list.append(float(line))

with open('loss_record/loss.txt', 'w') as f:
    for i in range(len(EL_Rec_list)):
        f.write("{:.5f}  {:.5f}  {:.5f}\n".format(DLRM_list[i], TT_Rec_list[i], EL_Rec_list[i]))


with open('loss_record/loss.txt') as f:
    DLRM_list = []
    TT_Rec_list = []
    EL_Rec_list = []
    for line in f:
        line_list = line.split()
        DLRM_list.append(float(line_list[0]))
        TT_Rec_list.append(float(line_list[1]))
        EL_Rec_list.append(float(line_list[2]))

    x = range(len(DLRM_list))


    plt.figure(figsize=(10,6), tight_layout=True)
    # #plotting
    plt.plot(x, DLRM_list, '-', linewidth=1)
    plt.plot(x, TT_Rec_list, '-', linewidth=1)
    plt.plot(x, EL_Rec_list, '-', linewidth=1)

    plt.ylim([0.3, 0.6])

    #customization
    plt.xlabel('Loss', font1)
    plt.ylabel('Iterations', font1)
    # plt.title('Rating troughtout the years')
    leg = plt.legend(title_fontsize = 13, labels=['DLRM', 'TT-Rec', 'EL-Rec'])

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.savefig('loss.png')