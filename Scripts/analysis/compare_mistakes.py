import torch 
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
grand_parentdir = os.path.dirname(currentdir)
sys.path.append(grand_parentdir)

from analysis import load_pickle, make_folder, LTCC, simple_indentifier, load_q_g, \
    compute_distance, misfit_ltcc, prcc_indentifier, PRCC, load_qs_qd_g, ToTensor, Image, save_image

if __name__ == "__main__":
    make_folder("Samples/")
    os.system("rm -rf Samples/*")
    
        
    dataset_name=  sys.argv[3]
    pickle_path1 = sys.argv[1]
    feature_dump1 = load_pickle(pickle_path1)

    pickle_path2 = sys.argv[2]
    feature_dump2 = load_pickle(pickle_path2)

    make_folder(f"Samples/WRONG_{pickle_path1}_CORRECT_{pickle_path2}/")
    make_folder(f"Samples/WRONG_{pickle_path1}_WRONG_{pickle_path2}/")


    # LTCC
    if dataset_name == "LTCC":
        dataset = LTCC(root=sys.argv[4])
        query_set = dataset.query
        indentifier_fn = simple_indentifier
        
        qf1, gf1, q_pids1, q_camids1, q_clothes_ids1, g_pids1, g_camids1, g_clothes_ids1 = load_q_g(feature_dump1)
        dismat1, _, _ = compute_distance(qf1, gf1, print)
        q_pids1, q_camids1, q_clothes_ids1 = q_pids1.numpy(), q_camids1.numpy(), q_clothes_ids1.numpy()
        g_pids1, g_camids1, g_clothes_ids1 = g_pids1.numpy(), g_camids1.numpy(), g_clothes_ids1.numpy()
        r1, misfits1, corrects1, count = misfit_ltcc(dismat1, g_pids1, q_pids1, g_camids1, q_camids1, g_clothes_ids1, q_clothes_ids1)

        qf2, gf2, q_pids2, q_camids2, q_clothes_ids2, g_pids2, g_camids2, g_clothes_ids2 = load_q_g(feature_dump2)
        dismat2, _, _ = compute_distance(qf2, gf2, print)
        q_pids2, q_camids2, q_clothes_ids2 = q_pids2.numpy(), q_camids2.numpy(), q_clothes_ids2.numpy()
        g_pids2, g_camids2, g_clothes_ids2 = g_pids2.numpy(), g_camids2.numpy(), g_clothes_ids2.numpy()
        r2, misfits2, corrects2, count = misfit_ltcc(dismat2, g_pids2, q_pids2, g_camids2, q_camids2, g_clothes_ids2, q_clothes_ids2)
        
    # PRCC
    elif dataset_name == "PRCC":
        assert False, "not yet verified"
        dataset = PRCC(root=sys.argv[3])
        query_set = dataset.query_diff
        indentifier_fn = prcc_indentifier
        
        qsf, qs_pids, qs_camids, qs_clothes_ids, \
            qdf, qd_pids, qd_camids, qd_clothes_ids, \
            gf, g_pids, g_camids, g_clothes_ids = load_qs_qd_g(feature_dump)

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
        
        r1, misfits, count = misfit_ltcc(distmat_diff, g_pids, qd_pids, g_camids, qd_camids, g_clothes_ids, qd_clothes_ids)

    print(r1, r1 / count)
    print(r2, r2 / count)
    fn = ToTensor()
    misfit1_dict = {e[0]: e[1] for e in misfits1}
    misfit2_dict = {e[0]: e[1] for e in misfits2}

    correct1_dict = {e[0]: e[1] for e in corrects1}
    correct2_dict = {e[0]: e[1] for e in corrects2}
    
    c1_wrong_c2_correct = []
    c1_wrong_c2_correct = []

    root_w_c=f"Samples/WRONG_{pickle_path1}_CORRECT_{pickle_path2}/"
    root_w_w=f"Samples/WRONG_{pickle_path1}_WRONG_{pickle_path2}/"

    counter = 0 
    for key in misfit1_dict:
        counter += 1
        query = key
        query_path = query_set[query][0]
        query_identifier = indentifier_fn(query_path)
        query = Image.open(query_path).convert('RGB')
        query = query.resize((192,384))
        query = fn(query)

        gallery = misfit1_dict[key]
        gallery_path = dataset.gallery[gallery][0]
        gallery_identifier1 = indentifier_fn(gallery_path)
        gallery = Image.open( gallery_path ).convert('RGB')
        gallery = gallery.resize((192,384))
        gallery1 = fn(gallery)

        # imgs1 = torch.stack([query, gallery])

        if key in misfit2_dict:
            gallery = misfit2_dict[key]
            gallery_path = dataset.gallery[gallery][0]
            gallery_identifier2 = indentifier_fn(gallery_path)
            gallery = Image.open( gallery_path ).convert('RGB')
            gallery = gallery.resize((192,384))
            gallery2 = fn(gallery)
            # imgs2 = torch.stack([query, gallery])

            save_image(query, f"{root_w_w}/{counter}_Q-{query_identifier}_C1.png")
            save_image(gallery1, f"{root_w_w}/{counter}_G-{gallery_identifier1}_C1.png")
            save_image(gallery2, f"{root_w_w}/{counter}_G-{gallery_identifier2}_C2.png")
        else:
            gallery = correct2_dict[key]
            gallery_path = dataset.gallery[gallery][0]
            gallery_identifier2 = indentifier_fn(gallery_path)
            gallery = Image.open( gallery_path ).convert('RGB')
            gallery = gallery.resize((192,384))
            gallery2 = fn(gallery)
            # imgs2 = torch.stack([query, gallery])

            save_image(query, f"{root_w_c}/{counter}_Q-{query_identifier}_C1.png")
            save_image(gallery1, f"{root_w_c}/{counter}_G-{gallery_identifier1}_C1.png")
            save_image(gallery2, f"{root_w_c}/{counter}_G-{gallery_identifier2}_C2.png")

            
