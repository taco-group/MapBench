import numpy as np
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import json
import os
import cv2
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import logging
import multiprocessing
import argparse

def load_graph(file_path):
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    return G

def get_landmark(G):
    other_labels = ['Intersections', 'Adjacent']
    landmark_list = [data['label'] for _, data in G.nodes(data=True) if 'label' in data and data['label'] not in other_labels]
    return landmark_list

def landmark_preprocessing(G, landmark, similarity_model):
    landmark_list = get_landmark(G)
    similarity_list = []
    for mark in landmark_list:
        similarity_test_pair = [mark, landmark]
        sentence_embeddings = similarity_model.encode(similarity_test_pair)
        spot_landmark_similarity = cosine_similarity(sentence_embeddings)[0][1]
        similarity_list.append(spot_landmark_similarity)
    idx = np.argmax(similarity_list)
    return landmark_list[idx]

def find_landmark(G, name, similarity_model):
    for node, data in G.nodes(data=True):
        landmarks = landmark_preprocessing(G, data['label'], similarity_model)
        if name in landmarks:
            return node

def shortest_path(G, landmark1, landmark2, similarity_model):
    coor1 = find_landmark(G, landmark1, similarity_model)
    coor2 = find_landmark(G, landmark2, similarity_model)
    return nx.shortest_path(G, source=coor1, target=coor2)

def path_eval(G, nav, start, end, pattern_mode, similarity_model):
    landmark_list = get_landmark(G)
    directive_edges = nav
    start, end = landmark_preprocessing(G, start, similarity_model), landmark_preprocessing(G, end, similarity_model)

    gt_path = shortest_path(G, start, end, similarity_model)
    gt_dis = 0
    for i in range(len(gt_path) - 1):
        gt_dis += np.sqrt((gt_path[i][0] - gt_path[i + 1][0]) ** 2 + (gt_path[i][1] - gt_path[i + 1][1]) ** 2)

    failture_flag = 0
    project_path = []
    language_path = []
    path_dis = 0
    prev = start

    if len(directive_edges) == 0:
        failture_flag = -1
        return failture_flag, failture_flag, [], []

    for cnt, e in enumerate(directive_edges):
        if "(" in e and ")" not in e:
            e = e + ")"
        if pattern_mode == 0:
            pattern = r"^(.*?) -> (.*?) \((.*?)\)$"
        else:
            pattern = r"(?:\d+\.\s)?(.*?) -> ([^(]*)"

        m = re.match(pattern, e)
        try:
            landmark1, landmark2 = m.group(1), m.group(2)
        except AttributeError:
            failture_flag = -4
            return failture_flag, failture_flag, [], []
        landmark1 = landmark_preprocessing(G, landmark1, similarity_model)
        landmark2 = landmark_preprocessing(G, landmark2, similarity_model)

        if landmark1 not in landmark_list or landmark2 not in landmark_list:
            failture_flag = -2
            return failture_flag, failture_flag, [], []

        if prev != landmark1:
            failture_flag = -3
            return failture_flag, failture_flag, [], []
        prev = landmark2

        if cnt == 0 and landmark1 != start:
            failture_flag = -3
            return failture_flag, failture_flag, [], []
        if cnt == len(directive_edges) - 1 and landmark2 != end:
            failture_flag = -3
            return failture_flag, failture_flag, [], []

        coor1 = find_landmark(G, landmark1, similarity_model)
        coor2 = find_landmark(G, landmark2, similarity_model)
        language_path.extend((coor1, coor2))

        path_nodes = nx.shortest_path(G, source=coor1, target=coor2)
        project_path.extend(path_nodes)
        for i in range(len(path_nodes) - 1):
            path_dis += np.sqrt((path_nodes[i][0] - path_nodes[i + 1][0]) ** 2 + (path_nodes[i][1] - path_nodes[i + 1][1]) ** 2)

    path_score = path_dis / gt_dis
    failture_flag = 1
    return failture_flag, path_score, language_path, project_path

def main(args):
    log_file = os.path.join(args.log_dir, args.result_file.split('/')[-1])
    res_data = []
    with open(args.result_file, 'r', encoding='utf-8') as f:
        for line in f:
            res_data.append(json.loads(line))

    average_score = 0
    query_success_num = 0

    for query in res_data:
        saved_log = query
        start = query["start"]
        destination = query["destination"]
        answer = query["answer"]
        map_class = query["map_class"]
        image_id = query["image_id"]

        pkl_file = os.path.join("./pkl", map_class, f"{image_id}.pkl")
        G = load_graph(pkl_file)

        similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        failture_flag, path_score, language_path, project_path = path_eval(G, answer, start, destination, 0, similarity_model)

        if failture_flag <= 0:
            saved_log["success"] = 0
            saved_log["path_score"] = path_score

        else:
            saved_log["success"] = 1
            saved_log["path_score"] = path_score
            query_success_num += 1
            average_score += path_score

        with open(log_file,"a") as f:
            f.write(json.dumps(saved_log))
            f.write("\n")

    if query_success_num == 0:
            average_score = 0
    else:
        average_score = average_score / query_success_num

    print(f"Map: {map_class}, Success Query Num: {query_success_num}, Average Score: {average_score}, ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    main(args)