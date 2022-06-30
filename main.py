from PIL import Image
from distutils.log import info
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse
from utils.common import read_yaml


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    # initializing resnet for face img to embeding conversion
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 

    # photos folder path 
    path=config['path']['directory']
    file_emb=config['path']['save_info']
    img_path=config['img_path']['path']

    dataset=datasets.ImageFolder(path) 

    # accessing names of peoples from folder names
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 


    #def collate_fn needed for loader
    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    # list of cropped faces from photos folder
    face_list = [] 
    # list of names corrospoing to cropped photos
    name_list = [] 
    # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
    embedding_list = [] 

    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True) 
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
            name_list.append(idx_to_class[idx]) # names are stored in a list


    #Saving the info
    data = [embedding_list, name_list]
    torch.save(data, file_emb) # saving data.pt file

    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load(file_emb) # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="./config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        result =  main(config_path=parsed_args.config)
        print('Face matched with: ',result[0], 'With distance: ',result[1])

    except Exception as e:
        raise e