#import git

#repo_url = "https://github.com/jahongir7174/YOLOv8-human.git"
#destination_folder = "yolov8"
#repo = git.Repo.clone_from(repo_url, destination_folder)


#Téléchargement du modèle YOLOv7
import requests
url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
destination_file = "yolov7.pt"
response = requests.get(url)
with open(destination_file, "wb") as file:
   file.write(response.content)
print("Download completed!")


import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, xyxy2xywh, set_logging, increment_path, \
   scale_coords#, check_imshow, strip_optimizer, apply_classifier
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import csv  # Importez le module pour travailler avec les fichiers CSV
import os   # Importez le module os pour travailler avec les répertoires
import psutil  # Importez le module psutil pour mesurer l'utilisation du CPU et de la mémoire


random.seed(1)
def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    # Mesurer l'utilisation du CPU et de la mémoire avant le traitement des vidéos
    start_cpu_percent = psutil.cpu_percent(interval=None)
    start_memory_usage = psutil.virtual_memory().used
    
    # Vérifier si la source est une vidéo (supprimer les conditions liées aux images et flux)
    video_extensions = ('.mp4', '.avi', '.mov')  # Liste des extensions de fichiers vidéo
    is_video_source = any(source.lower().endswith(ext) for ext in video_extensions)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize

    set_logging()
    # Obtenir la liste des fichiers vidéo dans le répertoire source
    # source_path = Path(source)
    #video_files = [f for f in source_path.glob('*.mp4')]
    # Ouvrir le fichier CSV pour l'écriture
    csv_file = open('detection_results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Nom du fichier', 'Contient humain', 'Confiance maximale', 'Nombre d\'images humaines', 'Temps maximal par image', 'Temps minimal par image', 'Temps total du clip'])
    # Load model
    device = 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model,harger le modèle avec le GPU spécifié
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, opt.img_size)


    # Chargement des vidéos
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride) #gère les images ou les vidéos

    # Get names and colors
    names = ['person']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    print(f"Processing video")

    # Run inference
    t0 = time.time()
    startTime = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # convertir en float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference of 1 image
        t1 = time_synchronized()
        with torch.no_grad():   # OPTIMISATION : Utilisation de torch.no_grad() ,Calculating gradients would cause a GPU memory leak  
             pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Initialiser les variables pour les statistiques du clip
        total_time_per_clip = 0.0
        total_num_human_images = 0
        max_confidence = 0.0
        total_max_time_per_image = float('-inf')
        total_min_time_per_image = float('inf')
        # Initialize lists to store image processing times
        image_processing_times = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image : i est l'indice de l'image, et det est une liste de détections pour cette image
           
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Check if the "person" class is present in detections
               # if 0 in det[:, -1]:  # Assuming class index for "person" is 0
                #    total_num_human_images += 1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results : enregistrement des résultats dans un fichier de texte & ajout des boîtes englobantes
                for *xyxy, conf, cls in reversed(det):
                    #if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'#étiquette qui sera affichée sur l'image : nom de la classe + confiance
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1) #dessine une boîte englobante et l'étiquette associée sur l'image
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # calcule les coordonnées normalisées de la boîte englobante
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  #construit une ligne de texte contenant les informations sur la détection
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                 # Calculate maximum confidence for this image
                if det[:, 4].numel() > 0:
                     image_max_confidence = det[:, 4].max()
                     max_confidence = max(max_confidence, image_max_confidence)

                 # Check if the "person" class is present in detections
                person_detected = any(det[:, -1] == 0)  # Assuming class index for "person" is 0

                if person_detected:
                    total_num_human_images += 1  # Increment the count
 
           

             # Calculate time taken for processing this image
            image_processing_time = t2 - t1
            image_processing_times.append(image_processing_time)

            # Calculate min, max, and average processing times
            total_min_time_per_image = min(image_processing_times)
            total_max_time_per_image = max(image_processing_times)
            total_average_time_per_image = sum(image_processing_times) / len(image_processing_times)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if is_video_source:
               if vid_path != save_path:  # new video
                  vid_path = save_path
                  if isinstance(vid_writer, cv2.VideoWriter): # a video is being created
                      vid_writer.release()  # release previous video writer
                  if vid_cap:  # video
                      fps = vid_cap.get(cv2.CAP_PROP_FPS)
                      w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                      h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                  vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'),fps, (w, h))
               vid_writer.write(im0)

    # Calculate min, max, and average processing times
    #total_min_time_per_image = min(image_processing_times)
    #total_max_time_per_image = max(image_processing_times)
    #total_average_time_per_image = sum(image_processing_times) / len(image_processing_times)


           



    total_time_per_clip = (time.time() - t0)



    # Écrire la ligne CSV pour le clip actuel
    csv_writer.writerow([p.name, total_num_human_images > 0, max_confidence, total_num_human_images, total_max_time_per_image, total_min_time_per_image, total_time_per_clip])

     # Imprimer le message indiquant que la vidéo est terminée
    print(f'Done processing video')

    #time for video
    print(f'Total time processing ({time.time() - t0:.3f}s)')


    # Mesurer l'utilisation du CPU et de la mémoire après le traitement des vidéos
    end_cpu_percent = psutil.cpu_percent(interval=None)
    end_memory_usage = psutil.virtual_memory().used

    # Calculer les différences d'utilisation du CPU et de la mémoire
    cpu_usage = end_cpu_percent - start_cpu_percent
    memory_usage = end_memory_usage - start_memory_usage

    # Afficher les résultats
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage} bytes")  

    # Print the results
    print(f" Maximum Confidence: {max_confidence:.4f}")
    print(f"Number of Images with Person Detections: {total_num_human_images}")
    print(f"Minimum Processing Time: {total_min_time_per_image:.4f} seconds")
    print(f"Maximum Processing Time: {total_max_time_per_image:.4f} seconds")
    print(f"Average Processing Time: {total_average_time_per_image:.4f} seconds")  

    # Fermer le fichier CSV
    csv_file.close()
    print('Done.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class: --class 0 (person class)')
    #parser.add_argument('--classes', nargs='+', type=int,default=[0],help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args(['--weights', 'yolov7.pt', '--source', 'capCam.mp4', '--img-size', '640', '--conf-thres', '0.5', '--classes', 0])

    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad(): #OPTIMISATION
        detect(opt)
