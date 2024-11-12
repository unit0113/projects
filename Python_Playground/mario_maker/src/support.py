import pygame
from os import walk


def import_folder(path):
    surface_list = []

    for folder_name, subfolders, img_files in walk(path):
        for image_name in img_files:
            full_path = path + "/" + image_name
            surface_list.append(pygame.image.load(full_path).convert_alpha())

    return surface_list


def import_folder_dict(path):
    surface_dict = {}

    for folder_name, subfolders, img_files in walk(path):
        for image_name in img_files:
            full_path = path + "/" + image_name
            surface_dict[image_name.split(".")[0]] = pygame.image.load(
                full_path
            ).convert_alpha()

    return surface_dict
