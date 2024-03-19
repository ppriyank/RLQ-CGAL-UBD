# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


from matplotlib import lines
from matplotlib import patches
from matplotlib.patheffects import withStroke
import pickle

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN = "#2FC1D3"
LGREEN="#72ba00"
BLUE = "#0a84e3"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"
ORANGE = "#f8a600"
PINK="#f8a6e7"
YELLOW="#fdeb01"
path_effects = [withStroke(linewidth=10, foreground="white")]


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data

def gender_graph():
    fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BLUE, ORANGE]
    # width of the bars
    barWidth = 0.3
    # Choose the height of the blue bars
    males = [0.8994484514212983, 0.9274637318659329, 0.9345679012345679]
    # Choose the height of the cyan bars
    females = [0.44668587896253603, 0.6122448979591837, 0.6854304635761589]
    # The x position of bars
    r1 = np.arange(len(males))
    r2 = [x + barWidth for x in r1]
    print(r1,r2)
    # Create blue bars
    for r, y, color in zip([r1,r2], [males, females], COLORS):
        # ax.bar(r, y, width = barWidth, color=color, lw=1.5, ec="white", capsize=7, zorder=12,)
        ax.bar(r, y, width = barWidth, color=color, lw=1.5, ec="white", capsize=7, )
        # ax.plot(r, y, color=color, lw=6, zorder=12,)
        for x_pos,y_pos in zip(r,y):
                # y_pos = y_pos.__round__(2)
                y_pos = int( y_pos * 100 ) / 100  # 2.23
                ax.text(
                x_pos -0.12, y_pos + 0.03, y_pos, color='black', fontsize=20, ha="left", path_effects=path_effects
            ) 


    ax.yaxis.set_ticks([i * 0.1 for i in range(0, 11, 2)])
    ax.yaxis.set_ticklabels([i * 10 for i in range(0, 11, 2)])
    ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    ax.xaxis.set_ticks([ r + barWidth/2 for r in range(len(males))])
    ax.xaxis.set_ticklabels(['CAL', 'BM (Our)', 'RLQ (Our)'], fontsize=25)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # Add grid lines
    ax.grid(axis = "y", color="#A8BAC4", lw=1.2)
    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["left"].set_linewidth(5)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")
    # Set custom limits
    ax.set_ylim(0.2, 1)
    ax.set_xlim(-0.2, 2.5)
    PAD = 35 * 0.01
    for label in [i * 0.1 for i in range(0, 11, 2)]:
        label = label.__round__(2)
        print(label)
        ax.text(
            -.5, label , label, 
            ha="left", va="baseline", fontsize=20,
        )
    plt.savefig("gender.png")

def low_res():
    # X = ["Celeb HD", "OOF", "Pixelation", "Motion Blur"]
    X = ["CAL", "BM (Our)", "RLQ (Our)"]
    barWidth = 2

    X_pos = [10,20,30]
    CAL = [37.3, 30.3, 25.3, 21.9]
    BM = [35.7, 28.9, 23.6, 18.8]
    FINAL = [45.1, 39.9, 31.1, 31.0]
    percentages = [CAL, BM, FINAL]
    COLORS = [BLUE, YELLOW, LGREEN, PINK]
    N = len(FINAL)
    # Initialize plot ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    # Add lines with dots
    # Note the zorder to have dots be on top of the lines
    X_label = []

    for i,percentage in enumerate(percentages):
        r = [X_pos[i] + j * barWidth for j in range(N)]
        mean_x = sum(r) / N
        X_label += [mean_x]
        print(percentage, r, mean_x, [percentage[0] - x for x in percentage])

        # ax.bar(r, percentage, width = barWidth, color=COLORS, lw=1.5, ec="white", capsize=7, )
        ax.bar(r, percentage, width = barWidth, color=COLORS, lw=0.3, ec="black", capsize=7, )
        # ax.plot(X_pos, percentage, color=color, lw=5)
        # ax.scatter(X_pos, percentage, fc=color, s=100, lw=1.5, ec="white", zorder=12)


    ax.yaxis.set_ticks([i * 5 for i in range(3, 11)])
    ax.yaxis.set_ticklabels([i * 5 for i in range(3, 11
    )])
    ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    ax.xaxis.set_ticks(X_label)
    ax.xaxis.set_ticklabels(X, fontsize=25)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # Add grid lines
    ax.grid(axis = "y", color="#A8BAC4", lw=1.2)

    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    # # Set custom limits
    ax.set_ylim(10, 47)
    ax.set_xlim(8, 38)


    # Add labels for vertical grid lines -----------------------
    # The pad is equal to 1% of the vertical range (35 - 0)
    PAD = 35 * 0.01
    for label in [i * 5 for i in range(2, 10)]:
        ax.text(
            8, label + PAD, label, 
            ha="right", va="baseline", fontsize=20,
        )
    plt.savefig("Low_res.png")

def low_res2():
    X = ["Celeb HD", "OOF", "Pixelation", "Motion Blur"]
    X_pos = [10,20,30, 40]
    CAL = [37.3, 30.3, 25.3, 21.9]
    BM = [35.7, 28.9, 23.6, 18.8]
    FINAL = [45.1, 39.9, 31.1, 31.0]
    percentages = [CAL, FINAL, BM]
    COLORS = [BLUE, ORANGE, PINK]
    N = len(FINAL)
    # Initialize plot ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    # Add lines with dots
    # Note the zorder to have dots be on top of the lines
    
    for percentage, color in zip(percentages, COLORS):
        ax.plot(X_pos, percentage, color=color, lw=5)
        ax.scatter(X_pos, percentage, fc=color, s=100, lw=1.5, ec="white", zorder=12)


    ax.yaxis.set_ticks([i * 5 for i in range(3, 11)])
    ax.yaxis.set_ticklabels([i * 5 for i in range(3, 11
    )])
    ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    ax.xaxis.set_ticks(X_pos)
    ax.xaxis.set_ticklabels(X, fontsize=20)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # Add grid lines
    ax.grid(axis = "both", color="#A8BAC4", lw=1.2)

    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    # # Set custom limits
    ax.set_ylim(15, 47)
    ax.set_xlim(6, 42)

    # # Add labels for vertical grid lines -----------------------
    # # The pad is equal to 1% of the vertical range (35 - 0)
    PAD = 35 * 0.01
    for label in [i * 5 for i in range(3, 10)]:
        ax.text(
            7, label + PAD, label, 
            ha="right", va="baseline", fontsize=20,
        )
    plt.savefig("Low_res2.png")



def knn(LTCC, y_range, name):
    
    KNN=[5, 10, 15, 20, 25, 30, 35, 40 ] 
    
    
    COLORS = [BLUE, ORANGE, PINK]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Add lines with dots
    # Note the zorder to have dots be on top of the lines
    color = ORANGE
    ax.plot(KNN, LTCC, color=color, lw=5)
    ax.scatter(KNN, LTCC, fc=color, s=100, lw=1.5, ec="white", zorder=12)

    
    ax.yaxis.set_ticks(y_range[:-1])
    ax.yaxis.set_ticklabels(y_range[:-1])
    ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    ax.xaxis.set_ticks(KNN)
    ax.xaxis.set_ticklabels(KNN, fontsize=20)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # Add grid lines
    ax.grid(axis = "both", color="#A8BAC4", lw=1.2)

    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    # # Set custom limits
    ax.set_ylim(y_range[0], y_range[-1])
    ax.set_xlim(1, 41)

    # Add labels for vertical grid lines -----------------------
    # The pad is equal to 1% of the vertical range (35 - 0)
    PAD = 35 * 0.01
    for label in y_range[:-1]:
        ax.text(
            1, label + PAD, label, 
            ha="right", va="baseline", fontsize=20,
        )
    plt.savefig(f"{name}.png")

def distillation(name):
    x = load_pickle(name + '_tsne')
    N_HR = x['N_HR']
    hr_f = x['X'][:N_HR]
    lr_f = x['X'][ N_HR:]
    
    X_pos = range(-100, 110, 50)
    fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BLUE, RED]

    ax.scatter(hr_f[:,0], hr_f[:,1], fc=GREEN, s=20, zorder=12, ec="white", lw=0.25, alpha=1)
    ax.scatter(lr_f[:,0], lr_f[:,1], fc=RED, s=20, zorder=12, ec="white", lw=0.3, alpha=0.5)

        
    # BROWN = "#AD8C97"
    #  = "#7d3a46"
    #  = "#2FC1D3"
    # LGREEN="#72ba00"
    # BLUE = "#0a84e3"
    # RED = "#E3120B"
    # ORANGE = "#f8a600"
    # PINK="#f8a6e7"
    

    ax.yaxis.set_ticks(X_pos)
    ax.yaxis.set_ticklabels(X_pos, fontsize=20)
    # ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    ax.xaxis.set_ticks(X_pos)
    ax.xaxis.set_ticklabels(X_pos, fontsize=20)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # # Add grid lines
    ax.grid(axis = "both", color="#A8BAC4", lw=.5)

    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")

    plt.savefig(f"{name}.png")


def Dist_graph():
    fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BLUE, YELLOW, LGREEN, PINK]
    # width of the bars

    barWidth = 0.2

    X = load_pickle(f'BM_28_1_LTCC_Dist')
    BM_X_NEG_lr, BM_X_NEG_hr, BM_X_NEG_lr_hr, BM_X_POS_hr, BM_X_POS_lr, BM_X_POS_lr_hr =X['X_NEG_lr'], np.array(X['X_NEG_hr']), np.array(X['X_NEG_lr_hr']), np.array(X['X_POS_hr']), np.array(X['X_POS_lr']), np.array(X['X_POS_lr_hr'])
    
    X = load_pickle(f'BM_28_2_TS_LTCC_Dist')
    BM_UBD_X_NEG_lr, BM_UBD_X_NEG_hr, BM_UBD_X_NEG_lr_hr, BM_UBD_X_POS_hr, BM_UBD_X_POS_lr, BM_UBD_X_POS_lr_hr =X['X_NEG_lr'], np.array(X['X_NEG_hr']), np.array(X['X_NEG_lr_hr']), np.array(X['X_POS_hr']), np.array(X['X_POS_lr']), np.array(X['X_POS_lr_hr'])

    r1 = np.arange(2)
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2*barWidth for x in r1]
    print(r1,r2, r3)
    neg_lr_lr = [BM_X_NEG_lr.mean(), BM_UBD_X_NEG_lr.mean()]
    neg_hr_hr = [BM_X_NEG_hr.mean(), BM_UBD_X_NEG_hr.mean()]
    neg_lr_hr = [BM_X_NEG_lr_hr.mean(), BM_UBD_X_NEG_lr_hr.mean()]

    # Create blue bars
    for r, y, color in zip([r1,r2, r3], [neg_lr_lr, neg_hr_hr, neg_lr_hr], COLORS):
        # ax.bar(r, y, width = barWidth, color=color, lw=1.5, ec="white", capsize=7, zorder=12,)
        ax.bar(r, y, width = barWidth, color=color, lw=1.5, ec="white", capsize=7, )
        # ax.plot(r, y, color=color, lw=6, zorder=12,)
        # for x_pos,y_pos in zip(r,y):
        #         # y_pos = y_pos.__round__(2)
        #         y_pos = int( y_pos * 100 ) / 100  # 2.23
        #         ax.text(
        #         x_pos -0.12, y_pos + 0.03, y_pos, color='black', fontsize=20, ha="left", path_effects=path_effects
        #     ) 

    ax.yaxis.set_ticks([i * 0.2 for i in range(0, 10, 2)])
    ax.yaxis.set_ticklabels([i * 0.2 for i in range(0, 10, 2)])
    ax.yaxis.set_tick_params(labelleft=False, length=0)
    # Customize y-axis ticks

    import pdb
    pdb.set_trace()
    plt.savefig("temp.png")
    ax.xaxis.set_ticks([ r + barWidth/2 for r in range(len(males))])
    ax.xaxis.set_ticklabels(['CAL', 'BM (Our)', 'RLQ (Our)'], fontsize=25)
    ax.xaxis.set_tick_params(length=6, width=1.2)

    # # Make gridlines be below most artists.
    ax.set_axisbelow(True)

    # # Add grid lines
    ax.grid(axis = "y", color="#A8BAC4", lw=1.2)
    # Remove all spines but the one in the bottom
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["left"].set_linewidth(5)

    # Customize bottom spine
    ax.spines["bottom"].set_lw(1.2)
    ax.spines["bottom"].set_capstyle("butt")
    # Set custom limits
    ax.set_ylim(0.2, 1)
    ax.set_xlim(-0.2, 2.5)
    PAD = 35 * 0.01
    for label in [i * 0.1 for i in range(0, 11, 2)]:
        label = label.__round__(2)
        print(label)
        ax.text(
            -.5, label , label, 
            ha="left", va="baseline", fontsize=20,
        )
    # plt.savefig("gender.png")



# python Scripts/analysis/low_res_analysis_Celeb.py  Celeb_CAL_32_1_LR
# python Scripts/analysis/low_res_analysis_Celeb.py CAL_UBD_32_2_Celeb_HD CAL_UBD_32_2_Celeb_LR
# python Scripts/analysis/low_res_analysis_Celeb.py BM_28_1_TS_Celeb_HD BM_28_1_TS_Celeb_LR
# python Scripts/analysis/low_res_analysis_Celeb.py Celeb_Final_R_LA_15_B=32_1_HD Celeb_Final_R_LA_15_B=32_1_LQ 


# plt.clf()



# gender_graph()
# low_res()
# low_res2()
# knn(LTCC = [43.6, 45.4, 46.4, 44.1, 46.7, 43.9, 44.6, 44.1], y_range=[i * 2 for i in range(20, 25)], name="ltcc_knn")
# knn(LTCC = [61.9, 62.5, 64.0, 61.8, 63.4, 62.3, 63.1, 61.1], y_range=[i * 2 for i in range(29, 34)], name="prcc_knn")

# distillation('Celeb_CAL_32_1_HD')
# distillation('CAL_UBD_32_2_Celeb_HD')
# distillation('BM_28_1_TS_Celeb_HD')
# distillation('Celeb_Final_R_LA_15_B=32_1_HD')

Dist_graph()


# python Scripts/analysis/low_res_analysis_Celeb.py  BM_28_1_TS_Celeb_LR
# python Scripts/analysis/low_res_analysis_Celeb.py  Celeb_Final_R_LA_15_B=32_1_LQ 


# python Scripts/analysis/graph.py