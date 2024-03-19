import torch 
import os
import sys
import pandas as pd 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
grand_parentdir = os.path.dirname(currentdir)
sys.path.append(grand_parentdir)

from analysis import load_pickle, make_folder, LTCC, simple_indentifier, load_q_g, \
    compute_distance, misfit_ltcc, prcc_indentifier, PRCC, load_qs_qd_g, ToTensor, Image, save_image


import numpy as np
from sklearn.metrics import f1_score


if __name__ == "__main__":
    dataset_name=  sys.argv[2]
    pickle_path1 = sys.argv[1]
    feature_dump1 = load_pickle(pickle_path1)

    # LTCC
    if dataset_name == "LTCC":
        dataset = LTCC(root=sys.argv[3])
        query_set = dataset.query
        indentifier_fn = simple_indentifier
        qf1, gf1, q_pids1, q_camids1, q_clothes_ids1, g_pids1, g_camids1, g_clothes_ids1 = load_q_g(feature_dump1)
        dismat1, _, _ = compute_distance(qf1, gf1, print)
        q_pids1, q_camids1, q_clothes_ids1 = q_pids1.numpy(), q_camids1.numpy(), q_clothes_ids1.numpy()
        g_pids1, g_camids1, g_clothes_ids1 = g_pids1.numpy(), g_camids1.numpy(), g_clothes_ids1.numpy()
        r1, misfits1, corrects1, count = misfit_ltcc(dismat1, g_pids1, q_pids1, g_camids1, q_camids1, g_clothes_ids1, q_clothes_ids1)

        test_gender = pd.read_csv("Scripts/Helper/LTCC_test_Gender.csv")
        query_gender = test_gender[test_gender.Category == "query"]
        gallery_gender = test_gender[test_gender.Category == "test"]
        
    # PRCC
    elif dataset_name == "PRCC":
        dataset = PRCC(root=sys.argv[3])
        query_set = dataset.query_diff
        indentifier_fn = prcc_indentifier
        
        qsf, qs_pids, qs_camids, qs_clothes_ids, \
            qdf, qd_pids, qd_camids, qd_clothes_ids, \
            gf, g_pids, g_camids, g_clothes_ids = load_qs_qd_g(feature_dump1)

        m, n, k = qsf.shape[0], qdf.shape[0], gf.shape[0]
        distmat_same = torch.zeros((m, k))
        distmat_diff = torch.zeros((n, k))
        
        qsf, qdf, gf = torch.tensor(qsf).cuda(), torch.tensor(qdf).cuda(), torch.tensor(gf).cuda()
        # Cosine similarity
        for i in range(m):
            distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
        for i in range(n):
            distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
        distmat_same = distmat_same.numpy()
        distmat_diff = distmat_diff.numpy()
        
        r1, misfits1, corrects1, count = misfit_ltcc(distmat_diff, g_pids, qd_pids, g_camids, qd_camids, g_clothes_ids, qd_clothes_ids)
        test_gender = pd.read_csv("Scripts/Helper/PRCC_test_Gender.csv")
        query_gender = test_gender
        gallery_gender = test_gender

    print(r1, r1 / count)
    
    fn = ToTensor()
    male_female = 0
    male_male = 0
    female_male = 0
    female_female = 0 
    for ele in misfits1 :
        query_path = query_set[ele[0]][0]
        query_identifier = indentifier_fn(query_path)
        person_id = query_identifier.split("_")[0]
        q_gender = query_gender[query_gender.ID == int(person_id)].Gender.item()
        # query = Image.open(query_path).convert('RGB')
        # query = query.resize((192,384))
        # query = fn(query)
        # save_image(query, f"temp.png")
        
    
        gallery_path = dataset.gallery[ele[1]][0]
        gallery_identifier1 = indentifier_fn(gallery_path)
        person_id = gallery_identifier1.split("_")[0]
        g_gender = gallery_gender[gallery_gender.ID == int(person_id)].Gender.item()
        # gallery = Image.open( gallery_path ).convert('RGB')
        # gallery = gallery.resize((192,384))
        # gallery1 = fn(gallery)

        if q_gender == g_gender:
            if q_gender == 1: 
                male_male +=1
            else: 
                female_female+=1
        elif q_gender == 1: 
            male_female +=1
        else: 
            female_male+=1
    
    print(f"male_male = {male_male}")    
    print(f"male_female = {male_female}")    
    print(f"female_male = {female_male}")    
    print(f"female_female = {female_female}")    
    actual = np.repeat([1, 0], repeats=[male_male + male_female, female_male + female_female])
    pred = np.repeat([1, 0, 1, 0], repeats=[male_male, male_female, female_male, female_female]) 
    print(f" (Male) F1 mistakes + Correct = {f1_score(actual, pred)}")    
    pred = np.repeat([1, 0, 1, 0], repeats=[female_female, female_male, male_female, male_male]) 
    print(f" (Female) F1 mistakes + Correct = {f1_score(actual, pred)}")    

    for ele in corrects1:
        query_path = query_set[ele[0]][0]
        query_identifier = indentifier_fn(query_path)
        person_id = query_identifier.split("_")[0]
        q_gender = query_gender[query_gender.ID == int(person_id)].Gender.item()
        
        gallery_path = dataset.gallery[ele[1]][0]
        gallery_identifier1 = indentifier_fn(gallery_path)
        person_id = gallery_identifier1.split("_")[0]
        g_gender = gallery_gender[gallery_gender.ID == int(person_id)].Gender.item()

        assert q_gender == g_gender
        if q_gender == 1: 
            male_male +=1
        else: 
            female_female+=1
        
    print(f"male_male = {male_male}")    
    print(f"male_female = {male_female}")    
    print(f"female_male = {female_male}")    
    print(f"female_female = {female_female}")    
    actual = np.repeat([1, 0], repeats=[male_male + male_female, female_male + female_female])
    pred = np.repeat([1, 0, 1, 0], repeats=[male_male, male_female, female_male, female_female]) 
    print(f" (Male) F1 mistakes + Correct = {f1_score(actual, pred)}")    
    pred = np.repeat([1, 0, 1, 0], repeats=[female_female, female_male, male_female, male_male]) 
    print(f" (Female) F1 mistakes + Correct = {f1_score(actual, pred)}")    


# python Scripts/analysis/gender_res_analysis.py LT_CAL_32_1 LTCC /data/priyank/synthetic/LTCC/
# 162 0.413265306122449
# male_male = 114
# male_female = 13
# female_male = 16
# female_female = 87
#  (Male) F1 mistakes + Correct = 0.8871595330739299
#  (Female) F1 mistakes + Correct = 0.881057268722467
# male_male = 202
# male_female = 13
# female_male = 16
# female_female = 161
#  (Male) F1 mistakes + Correct = 0.9330254041570438
#  (Female) F1 mistakes + Correct = 0.8946015424164524

# python Scripts/analysis/gender_res_analysis.py LT_BM_28_2 LTCC /data/priyank/synthetic/LTCC/
# 165 0.42091836734693877
# male_male = 120
# male_female = 2
# female_male = 15
# female_female = 90
#  (Male) F1 mistakes + Correct = 0.933852140077821
#  (Female) F1 mistakes + Correct = 0.8598130841121495
# male_male = 213
# male_female = 2
# female_male = 15
# female_female = 162
#  (Male) F1 mistakes + Correct = 0.9616252821670429
#  (Female) F1 mistakes + Correct = 0.8654353562005278

# python Scripts/analysis/gender_res_analysis.py Final_R_LA_15_B=32_1 LTCC /data/priyank/synthetic/LTCC/
# 182 0.4642857142857143
# male_male = 112
# male_female = 6
# female_male = 7
# female_female = 85
#  (Male) F1 mistakes + Correct = 0.9451476793248945
#  (Female) F1 mistakes + Correct = 0.8708133971291866
# male_male = 209
# male_female = 6
# female_male = 7
# female_female = 170
#  (Male) F1 mistakes + Correct = 0.9698375870069605
#  (Female) F1 mistakes + Correct = 0.9002557544757033

# python Scripts/analysis/gender_res_analysis.py PR_CAL_32_1 PRCC /data/priyank/synthetic/PRCC/
# 1954 0.5515100197572679
# male_male = 1060
# male_female = 18
# female_male = 219
# female_female = 292
#  (Male) F1 mistakes + Correct = 0.8994484514212983
#  (Female) F1 mistakes + Correct = 0.44668587896253603
# male_male = 2154
# male_female = 18
# female_male = 219
# female_female = 1152
#  (Male) F1 mistakes + Correct = 0.947854785478548
#  (Female) F1 mistakes + Correct = 0.7001795332136446

# python Scripts/analysis/gender_res_analysis.py PR_BM_32_1 PRCC /data/priyank/synthetic/PRCC/
# 2095 0.5913068021450748
# male_male = 927
# male_female = 59
# female_male = 86
# female_female = 376
#  (Male) F1 mistakes + Correct = 0.9274637318659329
#  (Female) F1 mistakes + Correct = 0.6122448979591837
# male_male = 2113
# male_female = 59
# female_male = 86
# female_female = 1285
#  (Male) F1 mistakes + Correct = 0.9668268130862503
#  (Female) F1 mistakes + Correct = 0.7645051194539249

# python Scripts/analysis/gender_res_analysis.py Final_PR_R_LA_15_B=32_1 PRCC /data/priyank/synthetic/PRCC/
# 2303 0.65001411233418
# male_male = 757
# male_female = 37
# female_male = 69
# female_female = 377
#  (Male) F1 mistakes + Correct = 0.9345679012345679
#  (Female) F1 mistakes + Correct = 0.6854304635761589
# male_male = 2135
# male_female = 37
# female_male = 69
# female_female = 1302
#  (Male) F1 mistakes + Correct = 0.9757769652650823
#  (Female) F1 mistakes + Correct = 0.7627456565081173

# Male query --> female gallery 1 --> 5% --> 4$
# female query --> gallery male 42% --> 18% --> 14%


