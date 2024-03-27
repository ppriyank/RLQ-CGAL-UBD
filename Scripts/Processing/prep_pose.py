import os 
from PIL import Image
import numpy as np 
import random 

import pandas as pd 

import math 

import cv2
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
    
resize_dim=(192,384)    
l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 
    return 


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data

def draw_line(coordinates):
    part_line = {}
    for i in range(coordinates.shape[0]):
        part_line[i] = (int(coordinates[i][0])), int(coordinates[i][1])
    return part_line

def draw_circles(rgb_image, coordinates, name="temp.png", part_line={}, l_pair=None, draw_without_line=None, rgb_opacity=1, save_as_png=None):
    img = img.astype(np.uint8)
    for i in range(coordinates.shape[0]):
        cv2.circle(img, coordinates[i].astype(int), radius=2, color=(255,255,255), thickness=2)
        cv2.circle(img, coordinates[i].astype(int), radius=4, color=(0,0,0), thickness=1)
    # Image.fromarray(img).save("temp2.png")
    if not draw_without_line:
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                # X = (start_xy[0], end_xy[0])
                # Y = (start_xy[1], end_xy[1])
                # mX = np.mean(X)
                # mY = np.mean(Y)
                # length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                # angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                # stickwidth = 1.5
                # polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                # cv2.line(img, start_xy, end_xy, (255,255,255), 1)
                cv2.line(img, start_xy, end_xy, (255,0,0), 1)
    img.save(name)

def draw_circles_on_png(rgb_image, coordinates, name="temp.png", part_line={}, l_pair=None, draw_without_line=None, rgb_opacity=1, save_as_png=None):
    img = (np.zeros_like(rgb_image)+ 255).astype(np.uint8)
    
    for i in range(coordinates.shape[0]):
        cv2.circle(img, coordinates[i].astype(int), radius=2, color=(0,0,0), thickness=2)
        cv2.circle(img, coordinates[i].astype(int), radius=4, color=(0,0,0), thickness=1)
    
    if not draw_without_line:
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, (255,0,0), 1)
    img = Image.fromarray(img)
    #img.save("temp2.png")
    img = img.convert('RGBA')

    datas = img.getdata()
    new_data = []
    newData = []
    for item in datas:
        # change all white (also shades of whites)
        # pixels to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    # Save new image
    img.save(name, "PNG")
    # img.save(name)


def resize_coordinates(x, resize_dim, original_dim):
    return (x * (resize_dim / original_dim)).astype(int)

def vectroize_graphs(coordinates, l_pair=None, only_angle=True):
    length_vector = [] 
    angle_vector = [] 
    part_line = draw_line(coordinates) # ==> coordinates // 1
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            length_vector.append(length)
            angle_vector.append(angle)
    return length_vector, angle_vector  

def draw_image(rgb_image, coordinates, img_name, l_pair, resize_dim , resize_cordinate, dataset, draw_without_line=None, rgb_opacity=1, only_resize=None, save_as_png=None):
    # rgb_image.save("temp.png")
    # coordinates : (17,2)
    DRAW = draw_circles
    if save_as_png:
        DRAW = draw_circles_on_png
    if not only_resize:
        DRAW(rgb_image, coordinates, name=f"Samples/{dataset}_{img_name[:-4]}_orig.png", part_line=draw_line(coordinates), l_pair=l_pair, draw_without_line=draw_without_line, rgb_opacity=rgb_opacity, save_as_png=save_as_png)
    rgb_image = rgb_image.resize((resize_dim))
    DRAW(rgb_image, resize_cordinate, name=f"Samples/{dataset}_{img_name[:-4]}_resize.png", part_line=draw_line(resize_cordinate), l_pair=l_pair, draw_without_line=draw_without_line, rgb_opacity=rgb_opacity, save_as_png=save_as_png)
    

def vectorize_coorindates(coordinates, resize_cordinate, normalized_cood, l_pair):
    len_vector, ang_vector  = vectroize_graphs(coordinates, l_pair=l_pair)
    len_resized_vector, ang_resized_vector = vectroize_graphs(resize_cordinate, l_pair=l_pair)
    len_norm_vector, ang_norm_vector = vectroize_graphs(normalized_cood, l_pair=l_pair)

    Resized_Vec = np.array(len_resized_vector + ang_resized_vector)
    Resized_Vec_Cooridnates = np.concatenate([resize_cordinate.reshape(-1), Resized_Vec] )
    Resized_Only_Angle = np.array(ang_resized_vector)
    
    Norm_Vec = np.array(len_norm_vector + ang_norm_vector)
    Norm_Vec_Cooridnates = np.concatenate([normalized_cood.reshape(-1), Norm_Vec] )
    Norm_Only_Angle = np.array(ang_norm_vector)

    return Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle
            

def process_coordinates(pickle_path, resize_dim, original_dim):
    img = load_pickle( pickle_path )
    scores = img["yolo_scores"]
    if scores is None :
        return None, None, None 
    index = scores.argmax()

    coordinates = img["coordinates"][index]
    # feat = img["feat"][index]

    normalized_cood = img["normalized_coords"]
    if normalized_cood.shape != (17,2):
        normalized_cood = normalized_cood[index]
    # print(normalized_cood.shape)
    # (17,2)
            
    cor_x, cor_y = coordinates[:, 0].astype(int), coordinates[:, 1].astype(int)

    cor_x = resize_coordinates(cor_x, resize_dim[0], original_dim[0])
    cor_y = resize_coordinates(cor_y, resize_dim[1], original_dim[1])
    resize_cordinate = np.column_stack((cor_x, cor_y))
    
    return coordinates, resize_cordinate, normalized_cood








def clustering_X(X, i , Null_Label):
    kmeans = KMeans(n_clusters=i, random_state=0, max_iter=1000).fit(X)
    intertia = kmeans.inertia_
    X = kmeans.predict(X)
    X = np.concatenate([X +1, Null_Label])
    return X , intertia

def knee(X , name):
    plt.plot(X,R_LA)
    plt.savefig(name)
    plt.clf()

def clustering(Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list, 
            Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list, 
            n_cluster, df, Null_Label, dataset):

    R_LA, R_LAC, R_A = [], [], []
    N_LA, N_LAC, N_A = [], [], []
    for i in n_cluster:
        X_Resized_Vec, Resized_Vec_intertia = clustering_X(Resized_Vec_list, i , Null_Label)
        X_Resized_Vec_Cooridnates, Resized_Vec_Cooridnates_intertia = clustering_X(Resized_Vec_Cooridnates_list, i, Null_Label )
        X_Resized_Only_Angle, Resized_Only_Angle_intertia = clustering_X(Resized_Only_Angle_list, i, Null_Label )

        X_Norm_Vec, Norm_Vec_intertia = clustering_X(Norm_Vec_list, i, Null_Label )
        X_Norm_Vec_Cooridnates, Norm_Vec_Cooridnates_intertia = clustering_X(Norm_Vec_Cooridnates_list, i, Null_Label )
        X_Norm_Only_Angle, Norm_Only_Angle = clustering_X(Norm_Only_Angle_list, i, Null_Label )

        R_LA.append(Resized_Vec_intertia)
        R_LAC.append(Resized_Vec_Cooridnates_intertia)
        R_A.append(Resized_Only_Angle_intertia)

        N_LA.append(Norm_Vec_intertia)
        N_LAC.append(Norm_Vec_Cooridnates_intertia)
        N_A.append(Norm_Only_Angle)

        df[f"R_LA_{i}"] = X_Resized_Vec
        df[f"R_LAC_{i}"] = X_Resized_Vec_Cooridnates
        df[f"R_A_{i}"] = X_Resized_Only_Angle

        df[f"N_LA_{i}"] = X_Norm_Vec
        df[f"N_LAC_{i}"] = X_Norm_Vec_Cooridnates
        df[f"N_A_{i}"] = X_Norm_Only_Angle

    plt.plot(n_cluster,R_LA)
    plt.plot(n_cluster,R_LAC)
    plt.plot(n_cluster,R_A)
    plt.plot(n_cluster,N_LA)
    plt.plot(n_cluster,N_LAC)
    plt.plot(n_cluster,N_A)

    plt.savefig(f'{dataset}_knee_plt.png')

    return df
         
# LTCC & Celeb & DeepChange
def cluster_body_pose(root, src=None, dest=None, categories = ['train'], threshold=0.995, viz=None, n_cluster=-1, dataset=None, format="png", print_total_count=10):
    category = categories[0]
    images = os.listdir(os.path.join(src, category))
    
    image_names = []
    faulty = 0 
    
    Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list = [],[],[]
    Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list = [],[],[]
    faulty = []
    print_count = 0 
    for img_name in images:
        pickle_path = os.path.join(root, category, img_name[:-4])
        
        if not os.path.exists(pickle_path + ".pkl"):
            faulty.append(img_name[:-4])
        else:
            rgb_image = os.path.join(src, category, img_name[:-4] + f".{format}")
            rgb_image = Image.open(rgb_image).convert('RGB')
            width,height = rgb_image.size
            original_dim = (width,height)

            coordinates, resize_cordinate, normalized_cood = process_coordinates(pickle_path, resize_dim, original_dim)
            if coordinates is None:
                faulty.append(img_name[:-4])
                continue 
            image_names.append(img_name[:-4])
            if random.random() > threshold and viz:
                print_count += 1
                draw_image(rgb_image, coordinates, img_name, l_pair, resize_dim , resize_cordinate, dataset)
                # img_name = "WO_LINE_" + img_name
                # draw_image(rgb_image, coordinates, img_name, l_pair, resize_dim , resize_cordinate, dataset, draw_without_line=True)
                if print_count == print_total_count:
                    quit()
            
            Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle = vectorize_coorindates(coordinates, resize_cordinate, normalized_cood, l_pair)
            
            Resized_Vec_list.append(Resized_Vec)
            Resized_Vec_Cooridnates_list.append(Resized_Vec_Cooridnates)
            Resized_Only_Angle_list.append(Resized_Only_Angle)
            Norm_Vec_list.append(Norm_Vec)
            Norm_Vec_Cooridnates_list.append(Norm_Vec_Cooridnates)
            Norm_Only_Angle_list.append(Norm_Only_Angle)
            
    Resized_Vec_list = np.array(Resized_Vec_list)
    Resized_Vec_Cooridnates_list = np.array(Resized_Vec_Cooridnates_list)
    Resized_Only_Angle_list = np.array(Resized_Only_Angle_list)

    Norm_Vec_list = np.array(Norm_Vec_list)
    Norm_Vec_Cooridnates_list = np.array(Norm_Vec_Cooridnates_list)
    Norm_Only_Angle_list = np.array(Norm_Only_Angle_list)

    df = pd.DataFrame([], columns = ["Category" , "Image"]) 
    df["Image"] = image_names + faulty
    df["Category"] = category
    Null_Label = np.zeros(len(faulty))
    
    print(f"Faulty : {faulty}")
    return clustering(Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list, 
            Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list, 
            n_cluster, df, Null_Label, dataset)
    
# PRCC    
def cluster_body_pose2(root, src=None, dest=None, categories = ['train'], threshold=0.995, 
    n_cluster=-1, dataset=None, format="jpg", viz=None):
    category = categories[0]
    
    people_path = os.path.join(src, category)
    people = os.listdir(people_path)
    image_names = []
    faulty = 0 
    
    Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list = [],[],[]
    Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list = [],[],[]
    faulty = []
    for person in people:
        person_path = os.path.join(people_path, person)
        images = os.listdir(person_path)
        for img_name in images:
            pickle_path = os.path.join(root, category, person, img_name[:-4])
            
            if not os.path.exists(pickle_path + ".pkl"):
                faulty.append(person + "_" + img_name[:-4])
            else:
                
                rgb_image = os.path.join(person_path, img_name[:-4] + f".{format}")
                rgb_image = Image.open(rgb_image).convert('RGB')
                width,height = rgb_image.size
                original_dim = (width,height)

                coordinates, resize_cordinate, normalized_cood = process_coordinates(pickle_path, resize_dim, original_dim)
                if coordinates is None:
                    faulty.append(person + "_" + img_name[:-4])
                    continue 
                image_names.append(person + "_" + img_name[:-4])
                if random.random() > threshold and viz:
                    draw_image(rgb_image, coordinates, img_name, l_pair, resize_dim , resize_cordinate, dataset)
                    quit()
                
                Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle = vectorize_coorindates(coordinates, resize_cordinate, normalized_cood, l_pair)
                
                Resized_Vec_list.append(Resized_Vec)
                Resized_Vec_Cooridnates_list.append(Resized_Vec_Cooridnates)
                Resized_Only_Angle_list.append(Resized_Only_Angle)
                Norm_Vec_list.append(Norm_Vec)
                Norm_Vec_Cooridnates_list.append(Norm_Vec_Cooridnates)
                Norm_Only_Angle_list.append(Norm_Only_Angle)
                
    Resized_Vec_list = np.array(Resized_Vec_list)
    Resized_Vec_Cooridnates_list = np.array(Resized_Vec_Cooridnates_list)
    Resized_Only_Angle_list = np.array(Resized_Only_Angle_list)

    Norm_Vec_list = np.array(Norm_Vec_list)
    Norm_Vec_Cooridnates_list = np.array(Norm_Vec_Cooridnates_list)
    Norm_Only_Angle_list = np.array(Norm_Only_Angle_list)

    df = pd.DataFrame([], columns = ["Category" , "Image"]) 
    df["Image"] = image_names + faulty
    df["Category"] = category
    Null_Label = np.zeros(len(faulty))
    
    print(f"Faulty : {faulty}")
    return clustering(Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list, 
            Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list, 
            n_cluster, df, Null_Label, dataset)
    
# LTCC Test
def cluster_body_pose_test(root, src=None, dest=None, categories = ['train'], test=[], threshold=0.995, viz=None, n_cluster=-1, dataset=None, format="png", print_total_count=10):
    category = categories[0]
    images = os.listdir(os.path.join(src, category))
    image_names = []
    faulty = 0 
    Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list = [],[],[]
    Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list = [],[],[]
    faulty = []
    print_count = 0 

    def process_pickle(pickle_path, img_name, rgb_image):
        # rgb_image = os.path.join(src, category, img_name[:-4] + f".{format}")
        rgb_image = Image.open(rgb_image).convert('RGB')
        width,height = rgb_image.size
        original_dim = (width,height)

        coordinates, resize_cordinate, normalized_cood = process_coordinates(pickle_path, resize_dim, original_dim)
        if coordinates is None:
            faulty.append(img_name[:-4])
            return None
        
        Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle = vectorize_coorindates(coordinates, resize_cordinate, normalized_cood, l_pair)
        return Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle
            
    T_Resized_Vec_list, T_Resized_Vec_Cooridnates_list, T_Resized_Only_Angle_list = [],[],[]
    T_Norm_Vec_list, T_Norm_Vec_Cooridnates_list, T_Norm_Only_Angle_list = [],[],[]
    image_names = []
    faulty = []
    categories = []
    f_categories = []
    for t_folder in test:
        images = os.listdir(os.path.join(src, t_folder))
        for img_name in images:
            pickle_path = os.path.join(root, t_folder, img_name[:-4])
            name_id = f"{t_folder}/{img_name[:-4]}"
            if not os.path.exists(pickle_path + ".pkl"):
                faulty.append(name_id)
                f_categories.append(t_folder)
                continue 
            rgb_image = os.path.join(src, t_folder, img_name[:-4] + f".{format}")
            image_names.append( name_id )
            categories.append(t_folder)
            vecs = process_pickle(pickle_path, img_name, rgb_image)
            if vecs:
                Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, \
                    Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle = vecs
            else:
                continue 
            T_Resized_Vec_list.append(Resized_Vec)
            T_Resized_Vec_Cooridnates_list.append(Resized_Vec_Cooridnates)
            T_Resized_Only_Angle_list.append(Resized_Only_Angle)
            T_Norm_Vec_list.append(Norm_Vec)
            T_Norm_Vec_Cooridnates_list.append(Norm_Vec_Cooridnates)
            T_Norm_Only_Angle_list.append(Norm_Only_Angle)
    
    
    df = pd.DataFrame([], columns = ["Category" , "Image", "Pose_Vectors"]) 
    df["Image"] = image_names + faulty
    df["Category"] = categories + f_categories
    Null_Label = np.zeros(len(faulty))

    T_Resized_Vec_list = np.array(T_Resized_Vec_list)
    T_Resized_Vec_Cooridnates_list = np.array(T_Resized_Vec_Cooridnates_list)
    T_Resized_Only_Angle_list = np.array(T_Resized_Only_Angle_list)
    T_Norm_Vec_list = np.array(T_Norm_Vec_list)
    T_Norm_Vec_Cooridnates_list = np.array(T_Norm_Vec_Cooridnates_list)
    T_Norm_Only_Angle_list = np.array(T_Norm_Only_Angle_list)
    categories = np.array(categories)
    image_names = np.array(image_names)

    dist_Resized_Vec_list = ((T_Resized_Vec_list[categories == 'query'][:,None] - T_Resized_Vec_list[categories == 'test'])**2).sum(-1)
    dist_Resized_Vec_Cooridnates_list = ((T_Resized_Vec_Cooridnates_list[categories == 'query'][:,None] - T_Resized_Vec_Cooridnates_list[categories == 'test'])**2).sum(-1)
    dist_Resized_Only_Angle_list = ((T_Resized_Only_Angle_list[categories == 'query'][:,None] - T_Resized_Only_Angle_list[categories == 'test'])**2).sum(-1)
    dist_Norm_Vec_list = ((T_Norm_Vec_list[categories == 'query'][:,None] - T_Norm_Vec_list[categories == 'test'])**2).sum(-1)
    dist_Norm_Vec_Cooridnates_list = ((T_Norm_Vec_Cooridnates_list[categories == 'query'][:,None] - T_Norm_Vec_Cooridnates_list[categories == 'test'])**2).sum(-1)
    dist_Norm_Only_Angle_list = ((T_Norm_Only_Angle_list[categories == 'query'][:,None] - T_Norm_Only_Angle_list[categories == 'test'])**2).sum(-1)
    
    with open(f'Vec_list.pkl', 'wb') as handle:
        pickle.dump(dict(dist_Resized_Vec_list=dist_Resized_Vec_list, 
        T_Resized_Vec_list=T_Resized_Vec_list, 
        T_Resized_Vec_Cooridnates_list=T_Resized_Vec_Cooridnates_list, 
        T_Resized_Only_Angle_list=T_Resized_Only_Angle_list, 
        T_Norm_Vec_list=T_Norm_Vec_list, 
        T_Norm_Vec_Cooridnates_list=T_Norm_Vec_Cooridnates_list,
        T_Norm_Only_Angle_list=T_Norm_Only_Angle_list,
        image_names=image_names,
        ), handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    quit()
    

    df["Pose_Vectors"] = T_Resized_Vec_list + len(faulty) * [np.zeros_like(T_Resized_Vec_list[0])]
    print(f"Faulty : {faulty}")
    return clustering(T_Resized_Vec_list, T_Resized_Vec_Cooridnates_list, T_Resized_Only_Angle_list, T_Norm_Vec_list, T_Norm_Vec_Cooridnates_list, T_Norm_Only_Angle_list, n_cluster, df, Null_Label, dataset)
    
# PRCC Test     
def cluster_body_pose_test2(root, src=None, dest=None, categories = ['train'], threshold=0.995, 
    n_cluster=-1, dataset=None, format="jpg", viz=None):
    
    
    Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list = [],[],[]
    Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list = [],[],[]
    faulty = []
    categories = [] 
    f_categories = [] 
    image_names = []
    
    main_root = os.path.join(src, "test")
    for category in os.listdir( main_root ):
        people = os.listdir(os.path.join(main_root, category))
        for person in people:
            person_path = os.path.join(main_root, category, person)
            images = os.listdir(person_path)
            for img_name in images:
                pickle_path = os.path.join(root, "test", category, person, img_name[:-4])
                name_id = f"{category}/{person}_{img_name[:-4]}"
                if not os.path.exists(pickle_path + ".pkl"):
                    faulty.append(name_id)
                    f_categories.append(category)
                else:
                    rgb_image = os.path.join(person_path, img_name[:-4] + f".{format}")
                    rgb_image = Image.open(rgb_image).convert('RGB')
                    width,height = rgb_image.size
                    original_dim = (width,height)

                    coordinates, resize_cordinate, normalized_cood = process_coordinates(pickle_path, resize_dim, original_dim)
                    if coordinates is None:
                        faulty.append(name_id)
                        f_categories.append(category)
                        continue 
                    image_names.append(name_id)
                    categories.append(category)
                    Resized_Vec, Resized_Vec_Cooridnates, Resized_Only_Angle, Norm_Vec, Norm_Vec_Cooridnates, Norm_Only_Angle = vectorize_coorindates(coordinates, resize_cordinate, normalized_cood, l_pair)
                    
                    Resized_Vec_list.append(Resized_Vec)
                    Resized_Vec_Cooridnates_list.append(Resized_Vec_Cooridnates)
                    Resized_Only_Angle_list.append(Resized_Only_Angle)
                    Norm_Vec_list.append(Norm_Vec)
                    Norm_Vec_Cooridnates_list.append(Norm_Vec_Cooridnates)
                    Norm_Only_Angle_list.append(Norm_Only_Angle)
                    
    Resized_Vec_list = np.array(Resized_Vec_list)
    Resized_Vec_Cooridnates_list = np.array(Resized_Vec_Cooridnates_list)
    Resized_Only_Angle_list = np.array(Resized_Only_Angle_list)

    Norm_Vec_list = np.array(Norm_Vec_list)
    Norm_Vec_Cooridnates_list = np.array(Norm_Vec_Cooridnates_list)
    Norm_Only_Angle_list = np.array(Norm_Only_Angle_list)

    df = pd.DataFrame([], columns = ["Category" , "Image"]) 
    df["Image"] = image_names + faulty
    df["Category"] = categories + f_categories
    Null_Label = np.zeros(len(faulty))
    
    print(f"Faulty : {faulty}")
    return clustering(Resized_Vec_list, Resized_Vec_Cooridnates_list, Resized_Only_Angle_list, 
            Norm_Vec_list, Norm_Vec_Cooridnates_list, Norm_Only_Angle_list, 
            n_cluster, df, Null_Label, dataset)
  



if __name__ == "__main__":
    make_folder("Samples/")

    # LTCC
    # ltcc = "/home/c3-0/datasets/LTCC/LTCC_ReID/"
    # ltcc_pose="/home/c3-0/datasets/ID-Dataset/AlphaPose/ltcc/2DPose/"
    ltcc = "/data/priyank/synthetic/LTCC/LTCC_ReID/"
    ltcc_pose="/data/priyank/synthetic/LTCC/AlphaPose/2DPose/"
    dataset="LTCC"
    # df = cluster_body_pose(root=ltcc_pose, src=ltcc, categories = ['train'], threshold=0.8, viz=True, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df = cluster_body_pose_test(root=ltcc_pose, src=ltcc, categories = ['train'], test = ['query', 'test'], threshold=0.8, viz=True, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df = cluster_body_pose_test(root=ltcc_pose, src=ltcc, categories = ['train'], test = ['query', 'test'], threshold=0.8, viz=True, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df.to_csv(f"Scripts/Helper/{dataset}_TEST_Poses.csv", index=False,) 

    # PRCC
    prcc = "/data/priyank/synthetic/PRCC/rgb/"
    # prcc = "/home/c3-0/datasets/PRCC/prcc/rgb/"
    # prcc_pose="/home/c3-0/datasets/ID-Dataset/AlphaPose/PRCC/"
    prcc_pose = "/data/priyank/synthetic/PRCC/pose/"
    dataset="PRCC"
    # rsync -r '/home/c3-0/datasets/ID-Dataset/AlphaPose/PRCC/test' ucf2:/data/priyank/synthetic/PRCC/pose/
    # df = cluster_body_pose2(root=prcc_pose, src=prcc, categories = ['train'], threshold=0.5, viz=None, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df = cluster_body_pose_test2(root=prcc_pose, src=prcc, categories = ['train'], threshold=0.5, viz=None, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df.to_csv(f"Scripts/Helper/{dataset}_TEST_Pose_Cluster.csv", index=False,) 

    # Celeb-reID
    celeb="/home/c3-0/datasets/ID-Dataset/Celeb-reID/"
    celeb_pose="/home/c3-0/datasets/ID-Dataset/AlphaPose/Celeb-reID/"
    dataset="Celeb-reID"
    # df = cluster_body_pose(root=celeb_pose, src=celeb, categories = ['train'], threshold=0.5, viz=None, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset, format="jpg")
    # df.to_csv(f"Scripts/Helper/{dataset}_Pose_Cluster.csv", index=False,) 

    # Deep Clothes
    deepcchange="/home/c3-0/datasets/DeepChange/DeepChangeDataset/"
    deepcchange_pose="/home/c3-0/datasets/ID-Dataset/AlphaPose/DeepChangeDataset/"
    dataset="DeepChange"
    # df = cluster_body_pose(root=deepcchange_pose, src=deepcchange, categories = ['train-set'], threshold=0.5, viz=None, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset, format="jpg")
    # df.to_csv(f"Scripts/Helper/{dataset}_Pose_Cluster.csv", index=False,) 

    # LaST
    last="/home/c3-0/datasets/LaST/last/"
    last_pose="/home/c3-0/datasets/ID-Dataset/AlphaPose/LaST/"
    dataset="LaST"
    # df = cluster_body_pose2(root=last_pose, src=last, categories = ['train'], threshold=0.5, viz=None, n_cluster=[5,10,15,20,25,30,35,40], dataset=dataset)
    # df.to_csv(f"Scripts/Helper/{dataset}_Pose_Cluster.csv", index=False,) 


# srun --pty --qos=day --cpus-per-task=8 bash
# cd ~/CCReID/
# python Scripts/Processing/prep_pose.py