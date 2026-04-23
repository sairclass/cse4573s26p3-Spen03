'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []
    # [[box_1], [box_2], ..., [box_n]]

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # print(f'img size = {img.size}')
    # print(f'img shape= {img.shape}')
    # print(f'img = {img}')
    
    rgb = get_compatable_img(rgb_img_tensor= img) # convert img tensor to compatable numpy array
    face_locs = get_face_boxes(rgb= rgb, upsamples=1, hog= False)
    # print(f'face_locs = {face_locs}')

    for box_tup in face_locs:
        float_box_list = [float(i) for i in box_tup] # top, right, bottom, left
        top, right, bottom, left = float_box_list
        
        # print(f'top {top}, right {right}, bottom {bottom}, left {left}')
        
        topleft_x = left # (left = 0, right = +)
        topleft_y = top # (top = 0, bottom = +)

        box_width = right - left # (left = 0, right = +)
        box_height = bottom - top # (top = 0, bottom = +)

        formatted = [topleft_x, topleft_y, box_width, box_height]

        # detection_results.append(float_box_list)
        detection_results.append(formatted)

    print(f'detection_results {detection_results}')

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    
    # convert faces to encoding vectors
    my_face_vectors = [] # face encodings to be clustered
    my_vector_to_string = {} # the name of the image that the face encoding belongs to
    for im_str in imgs:
        # convert img to be compatable with face_recognition:
        my_img = get_compatable_img(rgb_img_tensor= imgs[im_str])

        box = get_face_boxes(rgb= my_img, upsamples=1, hog= False) # a list of the locations of faces in the image
            
        # a list of vector encodings for the faces in an image
        face_vectors = face_recognition.face_encodings(face_image= my_img, known_face_locations= box) 

        for vector in face_vectors:
            vector = torch.from_numpy(vector)
            my_face_vectors.append(vector)
            my_vector_to_string[vector] = im_str

    # cluster the vectors with k-means clustering
    my_vector_clusters = k_means_clustering(my_data= my_face_vectors, k_clusters= K, epsilon= 1e-4)

    # print(f'my_vector_clusters = {my_vector_clusters}')

    # convert the clustered vectors into the list of filenames that belong to the vectors.
    my_list_of_string_clusters = []
    for vector_cluster in my_vector_clusters:
        # print(f'vector_cluster = {vector_cluster}')
        my_list_of_strings = []
        for vect in vector_cluster:
            # print(f'vect = {vect}')
            my_string = my_vector_to_string[vect]
            # don't append the same image to the list (identical faces in the same image like a mirror or twin)
            if my_string not in my_list_of_strings:
                my_list_of_strings.append(my_string) 
            
        my_list_of_string_clusters.append(my_list_of_strings)
    
    cluster_results = my_list_of_string_clusters


    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def get_compatable_img(rgb_img_tensor):
    # convert the tensor to a numpy array that the library wants
    
    # convert from [C W H] tensor to [H W C] numpy
    return rgb_img_tensor.permute(1, 2, 0).numpy()
    

def get_face_boxes(rgb, upsamples=1, hog=True):
    # get face boxes the way face_recognition wants them formatted

    if hog == True:
        # returns [(top, right, bottom, left), ..., (top, right, bottom, left)]
        face_locs = face_recognition.face_locations(img= rgb, 
                                                    number_of_times_to_upsample= upsamples,
                                                    model= "hog") 
    else:
        # using CNN passses all tests but it is very slow
        # returns [(top, right, bottom, left), ..., (top, right, bottom, left)]
        face_locs = face_recognition.face_locations(img= rgb, 
                                                    number_of_times_to_upsample= upsamples,
                                                    model= "cnn") 

    
    
    
    

    return face_locs


def k_means_clustering(my_data, k_clusters, epsilon=1e-4):
    '''
    cluster data into k clusters
    '''
    # initialize
    rand_idxs = torch.randperm(len(my_data))[ : k_clusters]
    centroids = [my_data[idx] for idx in rand_idxs] # list of centroids, initially random datapoints from dataset
    converged = False

    while converged == False:
        clusters = [[] for k in range(k_clusters)]
        
        for point in my_data:
            # compare the distance of a point to every centroid
            my_centroid_idx = 0
            min_dist = torch.nn.functional.mse_loss(input= point, target= centroids[my_centroid_idx])
            for i in range(1, len(centroids)):
                d = torch.nn.functional.mse_loss(input= point, target= centroids[i])
                if d < min_dist:
                    min_dist = d
                    my_centroid_idx = i
            # the centroid closest to a point now owns that point
            clusters[my_centroid_idx].append(point)     
        
        # recalculate centroid positions:
        new_centroids = []
        within_threshold_count = 0
        for i, cluster in enumerate(clusters):
            # print(f'cluster{i} = {cluster}')
            new_centroids.append(calculate_centriod(cluster_list= cluster))
            # count how many centroids have converged
            if torch.nn.functional.mse_loss(input= new_centroids[i], target= centroids[i]) <= epsilon:
                within_threshold_count += 1

        # if all centroids have converged, stop algorithm 
        if within_threshold_count >= len(clusters):
            converged = True
            # break
        else:
            centroids = new_centroids # try again with updated centroids

    return clusters



def calculate_centriod(cluster_list):
    # find the mean of the points in a cluster
    # cluster_list is a list of tensors
    cluster_tensor = torch.stack(cluster_list)
    new_centroid_position = cluster_tensor.mean(dim=0)
    return new_centroid_position.detach()